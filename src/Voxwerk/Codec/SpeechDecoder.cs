using Voxwerk.IO;
using Voxwerk.Numerics;

namespace Voxwerk.Codec;

/// <summary>
/// Decoder half of the Qwen3-TTS 12 Hz speech tokenizer (codes → 24 kHz waveform): split RVQ
/// dequantization (1 semantic + 15 acoustic codebooks), causal pre-conv, an 8-layer sliding-window
/// transformer, 2 × (transposed-conv ×2 + ConvNeXt) upsampling, then a Snake/BigVGAN-style decoder
/// upsampling ×8·5·4·3 to 1920 samples per frame.
/// </summary>
public sealed unsafe class SpeechDecoder : IDisposable
{
    public const int SamplesPerFrame = 1920;
    public const int ChunkSize = 300;
    public const int LeftContext = 25;

    private readonly float[][] _codebooks;  // 16 × [2048, 256]
    private readonly Linear _firstOut, _restOut;
    private readonly Conv1d _preConv;
    private readonly Linear _inputProj, _outputProj;
    private readonly float[] _preNorm;
    private readonly CodecTransformer _transformer;
    private readonly (ConvTranspose1d Up, ConvNeXt Block)[] _upsample;
    private readonly Conv1d _decoderIn;
    private readonly DecoderBlock[] _blocks;
    private readonly Snake _finalAct;
    private readonly Conv1d _decoderOut;
    private readonly int _codebookDim;

    private sealed class ConvNeXt
    {
        public required float[] DwWeight, DwBias, NormW, NormB, Gamma;
        public required Linear Pw1, Pw2;
        public required int Dim;
    }

    private sealed class ResidualUnit
    {
        public required Snake Act1, Act2;
        public required Conv1d Conv1, Conv2;
    }

    private sealed class DecoderBlock
    {
        public required Snake Act;
        public required ConvTranspose1d Up;
        public required ResidualUnit[] Units;
    }

    public SpeechDecoder(SafeTensorsFile st)
    {
        const string p = "decoder.";
        int nq = 16;
        _codebooks = new float[nq][];
        for (int i = 0; i < nq; i++)
        {
            string cb = i == 0 ? $"{p}quantizer.rvq_first.vq.layers.0._codebook" : $"{p}quantizer.rvq_rest.vq.layers.{i - 1}._codebook";
            var sum = st.ReadFloat(cb + ".embedding_sum");
            var usage = st.ReadFloat(cb + ".cluster_usage");
            int dim = sum.Length / usage.Length;
            _codebookDim = dim;
            for (int r = 0; r < usage.Length; r++)
            {
                float inv = 1f / MathF.Max(usage[r], 1e-5f);
                for (int d = 0; d < dim; d++) sum[r * dim + d] *= inv;
            }
            _codebooks[i] = sum;
        }
        _firstOut = Linear.Load(st, p + "quantizer.rvq_first.output_proj");
        _restOut = Linear.Load(st, p + "quantizer.rvq_rest.output_proj");
        _preConv = Conv1d.Load(st, p + "pre_conv.conv");
        _inputProj = Linear.Load(st, p + "pre_transformer.input_proj");
        _outputProj = Linear.Load(st, p + "pre_transformer.output_proj");
        _preNorm = st.ReadFloat(p + "pre_transformer.norm.weight");
        _transformer = new CodecTransformer(st, p + "pre_transformer", layers: 8, hidden: 512, heads: 16, headDim: 64,
            intermediate: 1024, eps: 1e-5f, ropeTheta: 10000, window: 72, mimiStyle: false);

        _upsample = new (ConvTranspose1d, ConvNeXt)[2];
        for (int i = 0; i < 2; i++)
        {
            string u = $"{p}upsample.{i}.";
            int dim = st.Info(u + "1.norm.weight").Shape[0];
            _upsample[i] = (new ConvTranspose1d(st, u + "0.conv", stride: 2), new ConvNeXt
            {
                Dim = dim,
                DwWeight = st.ReadFloat(u + "1.dwconv.conv.weight"),
                DwBias = st.ReadFloat(u + "1.dwconv.conv.bias"),
                NormW = st.ReadFloat(u + "1.norm.weight"),
                NormB = st.ReadFloat(u + "1.norm.bias"),
                Gamma = st.ReadFloat(u + "1.gamma"),
                Pw1 = Linear.Load(st, u + "1.pwconv1"),
                Pw2 = Linear.Load(st, u + "1.pwconv2"),
            });
        }

        _decoderIn = Conv1d.Load(st, p + "decoder.0.conv");
        int[] rates = [8, 5, 4, 3];
        _blocks = new DecoderBlock[rates.Length];
        for (int i = 0; i < rates.Length; i++)
        {
            string b = $"{p}decoder.{i + 1}.block.";
            var units = new ResidualUnit[3];
            int[] dil = [1, 3, 9];
            for (int u = 0; u < 3; u++)
            {
                string r = $"{b}{u + 2}.";
                units[u] = new ResidualUnit
                {
                    Act1 = Snake.Load(st, r + "act1"),
                    Conv1 = Conv1d.Load(st, r + "conv1.conv", dilation: dil[u]),
                    Act2 = Snake.Load(st, r + "act2"),
                    Conv2 = Conv1d.Load(st, r + "conv2.conv"),
                };
            }
            _blocks[i] = new DecoderBlock { Act = Snake.Load(st, b + "0"), Up = new ConvTranspose1d(st, b + "1.conv", rates[i]), Units = units };
        }
        _finalAct = Snake.Load(st, p + $"decoder.{rates.Length + 1}");
        _decoderOut = Conv1d.Load(st, p + $"decoder.{rates.Length + 2}.conv");
    }

    /// <summary>Optional hook receiving named stage outputs [T, C] for parity tests.</summary>
    public Action<string, Activation>? Trace { get; set; }

    /// <summary>Decodes codec frames [frames][16] to a 24 kHz waveform of <c>frames × 1920</c> samples,
    /// in 300-frame chunks with 25 frames of left context (exactly like the reference <c>chunked_decode</c>).</summary>
    public float[] Decode(IReadOnlyList<int[]> codes, Parallelism par, CancellationToken ct = default)
    {
        var wav = new float[codes.Count * SamplesPerFrame];
        int start = 0;
        while (start < codes.Count)
        {
            ct.ThrowIfCancellationRequested();
            int end = Math.Min(start + ChunkSize, codes.Count);
            int context = start - LeftContext > 0 ? LeftContext : start;
            var chunk = new int[end - start + context][];
            for (int i = 0; i < chunk.Length; i++) chunk[i] = codes[start - context + i];
            var part = DecodeChunk(chunk, par);
            int skip = context * SamplesPerFrame;
            Array.Copy(part, skip, wav, start * SamplesPerFrame, (end - start) * SamplesPerFrame);
            start = end;
        }
        return wav;
    }

    /// <summary>Full-context decode of one chunk (no chunking).</summary>
    public float[] DecodeChunk(IReadOnlyList<int[]> codes, Parallelism par)
    {
        int T = codes.Count;
        int cd = _codebookDim;
        Activation x;
        // Split RVQ: semantic codebook → output_proj_first, acoustic codebooks summed → output_proj_rest.
        using (var q0 = new Activation(T, cd, zero: true))
        using (var q1 = new Activation(T, cd, zero: true))
        {
            for (int t = 0; t < T; t++)
            {
                var f = codes[t];
                if (f.Length != _codebooks.Length) throw new ArgumentException($"Each frame needs {_codebooks.Length} codes.");
                new ReadOnlySpan<float>(_codebooks[0], Math.Max(0, f[0]) * cd, cd).CopyTo(new Span<float>(q0.Row(t), cd));
                var acc = new Span<float>(q1.Row(t), cd);
                for (int g = 1; g < _codebooks.Length; g++)
                    System.Numerics.Tensors.TensorPrimitives.Add(acc, new ReadOnlySpan<float>(_codebooks[g], Math.Max(0, f[g]) * cd, cd), acc);
            }
            x = new Activation(T, _firstOut.Out);
            _firstOut.Forward(q0.Ptr, cd, T, x.Ptr, x.Channels, par);
            _restOut.Forward(q1.Ptr, cd, T, x.Ptr, x.Channels, par, accumulate: true);
        }
        Trace?.Invoke("dec_quantized", x);

        x = Replace(x, _preConv.ForwardCausal(x, PadMode.Zero, par));
        Trace?.Invoke("dec_pre_conv", x);

        using (var h = _inputProj.Forward(x, par))
        {
            _transformer.Forward(h, par);
            fixed (float* nw = _preNorm) Ops.RmsNorm(h.Ptr, h.Ptr, h.Time, h.Channels, nw, _transformer.Eps);
            x = Replace(x, _outputProj.Forward(h, par));
        }
        Trace?.Invoke("dec_pre_transformer", x);

        for (int i = 0; i < _upsample.Length; i++)
        {
            var (up, blk) = _upsample[i];
            x = Replace(x, up.Forward(x, par));
            ConvNeXtForward(blk, x, par);
            Trace?.Invoke($"dec_upsample{i}", x);
        }

        x = Replace(x, _decoderIn.ForwardCausal(x, PadMode.Zero, par));
        Trace?.Invoke("dec_decoder0", x);
        float[] wav = Trace != null || !Streaming ? VocoderFull(x, par) : VocoderStreaming(x, par);
        x.Dispose();
        System.Numerics.Tensors.TensorPrimitives.Min(wav, 1f, wav);
        System.Numerics.Tensors.TensorPrimitives.Max(wav, -1f, wav);
        return wav;
    }

    /// <summary>Run the audio-rate vocoder in causal segments (bit-identical to whole-sequence processing, but the
    /// working set stays cache-sized instead of streaming gigabytes of activations through memory).</summary>
    public bool Streaming { get; set; } = true;

    /// <summary>Rows (at the 50 Hz vocoder input rate) per streaming segment: 32 rows = 0.64 s of audio.</summary>
    public int StreamingSegment { get; set; } = 32;

    private float[] VocoderFull(Activation x, Parallelism par)
    {
        x = Clone(x);
        for (int i = 0; i < _blocks.Length; i++)
        {
            var b = _blocks[i];
            x = Replace(x, b.Up.Forward(x, par, pre: b.Act));
            foreach (var u in b.Units)
            {
                // x += conv2(snake2(conv1(snake1(x)))) with both activations fused into the conv inputs.
                using var c1 = u.Conv1.ForwardCausal(x, PadMode.Zero, par, pre: u.Act1);
                u.Act2.ApplyInPlace(c1, par);
                u.Conv2.ForwardCausal(c1, PadMode.Zero, par, accumulateInto: x);
            }
            Trace?.Invoke($"dec_decoder{i + 1}", x);
        }
        if (Trace != null)
        {
            using var act = new Activation(x.Time, x.Channels);
            _finalAct.Apply(x.Ptr, act.Ptr, x.Time, par);
            Trace("dec_decoder" + (_blocks.Length + 1), act);
        }
        using var y = _decoderOut.ForwardCausal(x, PadMode.Zero, par, pre: _finalAct);
        x.Dispose();
        return y.ToArray();
    }

    private float[] VocoderStreaming(Activation x, Parallelism par)
    {
        int upsample = 1;
        foreach (var b in _blocks) upsample *= b.Up.Stride;
        var wav = new float[(long)x.Time * upsample];
        // Per-layer causal history, zero-initialised (= the causal zero padding).
        var upState = _blocks.Select(b => new Activation(Math.Max(1, b.Up.History), b.Up.Cin, zero: true)).ToArray();
        var unitState = _blocks.Select(b => b.Units.Select(u => new Activation(Math.Max(1, u.Conv1.History), u.Conv1.Cin, zero: true)).ToArray()).ToArray();
        var outState = new Activation(Math.Max(1, _decoderOut.History), _decoderOut.Cin, zero: true);
        try
        {
            int seg = Math.Max(1, StreamingSegment);
            using var piece = new Activation(seg, x.Channels);
            for (int t0 = 0; t0 < x.Time; t0 += seg)
            {
                int n = Math.Min(seg, x.Time - t0);
                var h = n == seg ? piece : new Activation(n, x.Channels);
                Buffer.MemoryCopy(x.Row(t0), h.Ptr, (long)n * x.Channels * 4, (long)n * x.Channels * 4);
                for (int i = 0; i < _blocks.Length; i++)
                {
                    var b = _blocks[i];
                    var up = b.Up.ForwardStreaming(h, upState[i], par, pre: b.Act);
                    if (h != piece) h.Dispose();
                    h = up;
                    for (int u = 0; u < b.Units.Length; u++)
                    {
                        var unit = b.Units[u];
                        using var c1 = unit.Conv1.ForwardStreaming(h, unitState[i][u], par, pre: unit.Act1);
                        unit.Act2.ApplyInPlace(c1, par);
                        unit.Conv2.ForwardCausal(c1, PadMode.Zero, par, accumulateInto: h);
                    }
                }
                using (var y = _decoderOut.ForwardStreaming(h, outState, par, pre: _finalAct))
                    y.AsSpan().CopyTo(wav.AsSpan((int)((long)t0 * upsample), y.Time));
                if (h != piece) h.Dispose();
            }
        }
        finally
        {
            foreach (var s in upState) s.Dispose();
            foreach (var a in unitState) foreach (var s in a) s.Dispose();
            outState.Dispose();
        }
        return wav;
    }

    private static Activation Clone(Activation x)
    {
        var c = new Activation(x.Time, x.Channels);
        Buffer.MemoryCopy(x.Ptr, c.Ptr, x.Length * 4, x.Length * 4);
        return c;
    }

    private static Activation Replace(Activation old, Activation next)
    {
        old.Dispose();
        return next;
    }

    /// <summary>ConvNeXt block (in place): causal depthwise conv k7 → LayerNorm → Linear ×4 → GELU → Linear → γ → residual.</summary>
    private static void ConvNeXtForward(ConvNeXt b, Activation x, Parallelism par)
    {
        int T = x.Time, C = b.Dim, K = b.DwWeight.Length / C;
        using var dw = new Activation(T, C);
        fixed (float* w = b.DwWeight, bias = b.DwBias, nw = b.NormW, nb = b.NormB)
        {
            // weight [C, 1, K]; causal: y[t] = Σ_k w[k] · x[t − (K−1) + k]
            float* wp = w, bp = bias;
            par.ForRange(T, 64, (t0, t1) =>
            {
                for (int t = t0; t < t1; t++)
                {
                    float* o = dw.Row(t);
                    for (int c = 0; c < C; c++) o[c] = bp[c];
                    for (int k = 0; k < K; k++)
                    {
                        int src = t - (K - 1) + k;
                        if (src < 0) continue;
                        float* xi = x.Row(src);
                        for (int c = 0; c < C; c++) o[c] = MathF.FusedMultiplyAdd(wp[c * K + k], xi[c], o[c]);
                    }
                }
            });
            Ops.LayerNorm(dw.Ptr, dw.Ptr, T, C, nw, nb, 1e-6f);
        }
        using var h = b.Pw1.Forward(dw, par);
        Ops.Gelu(h.Ptr, h.Length);
        using var o2 = b.Pw2.Forward(h, par);
        for (int t = 0; t < T; t++)
            System.Numerics.Tensors.TensorPrimitives.FusedMultiplyAdd(
                new ReadOnlySpan<float>(o2.Row(t), C), b.Gamma, new ReadOnlySpan<float>(x.Row(t), C), new Span<float>(x.Row(t), C));
    }

    public void Dispose()
    {
        _firstOut.Dispose();
        _restOut.Dispose();
        _preConv.Dispose();
        _inputProj.Dispose();
        _outputProj.Dispose();
        _transformer.Dispose();
        foreach (var (u, b) in _upsample)
        {
            u.Dispose();
            b.Pw1.Dispose();
            b.Pw2.Dispose();
        }
        _decoderIn.Dispose();
        foreach (var b in _blocks)
        {
            b.Up.Dispose();
            foreach (var u in b.Units)
            {
                u.Conv1.Dispose();
                u.Conv2.Dispose();
            }
        }
        _decoderOut.Dispose();
    }
}
