using Voxwerk.IO;
using Voxwerk.Numerics;

namespace Voxwerk.Codec;

/// <summary>
/// Encoder half of the 12 Hz speech tokenizer (a Mimi encoder): SEANet causal conv stack (×4·5·6·8
/// downsampling of 24 kHz audio to 25 Hz), an 8-layer causal transformer, a stride-2 conv to 12.5 Hz, and
/// split residual vector quantization. Only the 16 codebooks the talker uses (1 semantic + 15 acoustic)
/// are evaluated.
/// </summary>
public sealed unsafe class SpeechEncoder : IDisposable
{
    public const int SampleRate = 24000;
    public const int SamplesPerFrame = 1920;
    public const int Codebooks = 16;

    private readonly List<(int Index, object Layer)> _seanet = new();
    private readonly CodecTransformer _transformer;
    private readonly Conv1d _downsample;
    private readonly Linear _semanticIn, _acousticIn;
    private readonly PackedMatrix[] _codebooks;      // for x·e
    private readonly float[][] _codebookRows;        // [2048, 256]
    private readonly float[][] _codebookNorms;       // ||e||²
    private readonly int _dim;

    private sealed class ResBlock
    {
        public required Conv1d Conv1, Conv2;
    }

    public SpeechEncoder(SafeTensorsFile st)
    {
        const string p = "encoder.";
        int[] ratios = [4, 5, 6, 8];
        int idx = 0;
        _seanet.Add((idx++, Conv1d.Load(st, $"{p}encoder.layers.0.conv")));
        foreach (int r in ratios)
        {
            _seanet.Add((idx, new ResBlock
            {
                Conv1 = Conv1d.Load(st, $"{p}encoder.layers.{idx}.block.1.conv"),
                Conv2 = Conv1d.Load(st, $"{p}encoder.layers.{idx}.block.3.conv"),
            }));
            idx++;
            _seanet.Add((idx++, "elu"));
            _seanet.Add((idx, Conv1d.Load(st, $"{p}encoder.layers.{idx}.conv", stride: r)));
            idx++;
        }
        _seanet.Add((idx++, "elu"));
        _seanet.Add((idx, Conv1d.Load(st, $"{p}encoder.layers.{idx}.conv")));

        _transformer = new CodecTransformer(st, p + "encoder_transformer", layers: 8, hidden: 512, heads: 8, headDim: 64,
            intermediate: 2048, eps: 1e-5f, ropeTheta: 10000, window: 0, mimiStyle: true);
        _downsample = Conv1d.Load(st, p + "downsample.conv", stride: 2, bias: false);
        _semanticIn = Linear.Load(st, p + "quantizer.semantic_residual_vector_quantizer.input_proj");
        _acousticIn = Linear.Load(st, p + "quantizer.acoustic_residual_vector_quantizer.input_proj");
        _codebooks = new PackedMatrix[Codebooks];
        _codebookRows = new float[Codebooks][];
        _codebookNorms = new float[Codebooks][];
        for (int i = 0; i < Codebooks; i++)
        {
            string cb = i == 0
                ? $"{p}quantizer.semantic_residual_vector_quantizer.layers.0.codebook"
                : $"{p}quantizer.acoustic_residual_vector_quantizer.layers.{i - 1}.codebook";
            var sum = st.ReadFloat(cb + ".embed_sum");
            var usage = st.ReadFloat(cb + ".cluster_usage");
            int n = usage.Length, d = sum.Length / n;
            _dim = d;
            var norms = new float[n];
            for (int r = 0; r < n; r++)
            {
                float inv = 1f / MathF.Max(usage[r], 1e-5f);
                double s2 = 0;
                for (int k = 0; k < d; k++)
                {
                    sum[r * d + k] *= inv;
                    s2 += (double)sum[r * d + k] * sum[r * d + k];
                }
                norms[r] = (float)s2;
            }
            _codebookRows[i] = sum;
            _codebookNorms[i] = norms;
            fixed (float* e = sum) _codebooks[i] = PackedMatrix.FromLinear(e, n, d);
        }
    }

    /// <summary>Optional hook receiving named stage outputs [T, C] for parity tests.</summary>
    public Action<string, Activation>? Trace { get; set; }

    /// <summary>Encodes a 24 kHz mono waveform to codec frames [ceil(n / 1920)][16].</summary>
    public int[][] Encode(ReadOnlySpan<float> wav24k, Parallelism par)
    {
        int n = wav24k.Length;
        if (n == 0) return [];
        var x = new Activation(n, 1);
        wav24k.CopyTo(x.AsSpan());
        foreach (var (i, layer) in _seanet)
        {
            switch (layer)
            {
                case Conv1d conv:
                    x = Replace(x, conv.ForwardCausal(x, PadMode.Zero, par));
                    break;
                case ResBlock rb:
                {
                    using var a = new Activation(x.Time, x.Channels);
                    Ops.Elu(x.Ptr, a.Ptr, x.Length);
                    using var h = rb.Conv1.ForwardCausal(a, PadMode.Zero, par);
                    Ops.Elu(h.Ptr, h.Ptr, h.Length);
                    using var o = rb.Conv2.ForwardCausal(h, PadMode.Zero, par);
                    System.Numerics.Tensors.TensorPrimitives.Add(x.AsSpan(), o.AsSpan(), x.AsSpan());
                    break;
                }
                default:
                    Ops.Elu(x.Ptr, x.Ptr, x.Length);
                    break;
            }
            Trace?.Invoke($"enc_seanet_l{i}", x);
        }
        _transformer.Forward(x, par);
        Trace?.Invoke("enc_transformer", x);
        x = Replace(x, _downsample.ForwardCausal(x, PadMode.Replicate, par));
        Trace?.Invoke("enc_downsample", x);

        int frames = Math.Min(x.Time, (n + SamplesPerFrame - 1) / SamplesPerFrame);
        var codes = new int[frames][];
        for (int f = 0; f < frames; f++) codes[f] = new int[Codebooks];
        using (var sem = new Activation(x.Time, _dim))
        using (var res = new Activation(x.Time, _dim))
        {
            _semanticIn.Forward(x.Ptr, x.Channels, x.Time, sem.Ptr, _dim, par);
            _acousticIn.Forward(x.Ptr, x.Channels, x.Time, res.Ptr, _dim, par);
            Quantize(0, sem, codes, frames, subtract: false, par);
            for (int q = 1; q < Codebooks; q++) Quantize(q, res, codes, frames, subtract: true, par);
        }
        x.Dispose();
        return codes;
    }

    /// <summary>Nearest codebook entry (Euclidean) per frame; optionally subtracts it from the residual.</summary>
    private void Quantize(int q, Activation x, int[][] codes, int frames, bool subtract, Parallelism par)
    {
        int n = _codebookNorms[q].Length;
        using var scores = new Activation(x.Time, n);
        Gemm.Multiply(x.Ptr, _dim, x.Time, _codebooks[q], scores.Ptr, n, par);
        var norms = _codebookNorms[q];
        for (int t = 0; t < x.Time; t++)
        {
            float* s = scores.Row(t);
            int best = 0;
            float bestD = float.PositiveInfinity;
            for (int j = 0; j < n; j++)
            {
                float d = norms[j] - 2f * s[j];
                if (d < bestD)
                {
                    bestD = d;
                    best = j;
                }
            }
            if (t < frames) codes[t][q] = best;
            if (subtract)
                System.Numerics.Tensors.TensorPrimitives.Subtract(new ReadOnlySpan<float>(x.Row(t), _dim),
                    new ReadOnlySpan<float>(_codebookRows[q], best * _dim, _dim), new Span<float>(x.Row(t), _dim));
        }
    }

    private static Activation Replace(Activation old, Activation next)
    {
        old.Dispose();
        return next;
    }

    public void Dispose()
    {
        foreach (var (_, l) in _seanet)
        {
            if (l is Conv1d c) c.Dispose();
            if (l is ResBlock rb)
            {
                rb.Conv1.Dispose();
                rb.Conv2.Dispose();
            }
        }
        _transformer.Dispose();
        _downsample.Dispose();
        _semanticIn.Dispose();
        _acousticIn.Dispose();
        foreach (var c in _codebooks) c.Dispose();
    }
}
