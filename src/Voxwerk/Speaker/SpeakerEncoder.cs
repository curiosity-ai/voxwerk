using Voxwerk.IO;
using Voxwerk.Numerics;

namespace Voxwerk.Speaker;

/// <summary>
/// The Base model's ECAPA-TDNN speaker encoder: 128-bin log-mel → TDNN → 3 × SE-Res2Net blocks
/// (dilations 2, 3, 4) → multi-layer feature aggregation → attentive statistics pooling → 1024-d x-vector.
/// All convolutions use PyTorch's <c>padding="same"</c> with reflect padding.
/// </summary>
public sealed unsafe class SpeakerEncoder : IDisposable
{
    private const int Scale = 8;

    private readonly Conv1d _block0;
    private readonly SeRes2Block[] _blocks;
    private readonly Conv1d _mfa;
    private readonly Conv1d _aspTdnn, _aspConv;
    private readonly Conv1d _fc;

    private sealed class SeRes2Block
    {
        public required Conv1d Tdnn1, Tdnn2, Se1, Se2;
        public required Conv1d[] Res2;
    }

    public SpeakerEncoder(SafeTensorsFile st, string prefix = "speaker_encoder")
    {
        int[] dilations = [2, 3, 4];
        _block0 = Conv1d.Load(st, $"{prefix}.blocks.0.conv");
        _blocks = new SeRes2Block[3];
        for (int i = 0; i < 3; i++)
        {
            string b = $"{prefix}.blocks.{i + 1}.";
            _blocks[i] = new SeRes2Block
            {
                Tdnn1 = Conv1d.Load(st, b + "tdnn1.conv"),
                Tdnn2 = Conv1d.Load(st, b + "tdnn2.conv"),
                Se1 = Conv1d.Load(st, b + "se_block.conv1"),
                Se2 = Conv1d.Load(st, b + "se_block.conv2"),
                Res2 = Enumerable.Range(0, Scale - 1).Select(j => Conv1d.Load(st, $"{b}res2net_block.blocks.{j}.conv", dilation: dilations[i])).ToArray(),
            };
        }
        _mfa = Conv1d.Load(st, $"{prefix}.mfa.conv");
        _aspTdnn = Conv1d.Load(st, $"{prefix}.asp.tdnn.conv");
        _aspConv = Conv1d.Load(st, $"{prefix}.asp.conv");
        _fc = Conv1d.Load(st, $"{prefix}.fc");
        EmbeddingDim = _fc.Cout;
    }

    public int EmbeddingDim { get; }

    public Action<string, Activation>? Trace { get; set; }

    private static Activation TdnnRelu(Conv1d conv, Activation x, Parallelism par)
    {
        var y = conv.ForwardSame(x, PadMode.Reflect, par);
        Relu(y.AsSpan());
        return y;
    }

    private static void Relu(Span<float> x) => System.Numerics.Tensors.TensorPrimitives.Max(x, 0f, x);

    /// <summary>Computes the x-vector from a log-mel spectrogram [frames, 128].</summary>
    public float[] Forward(float[] mel, int frames, Parallelism par)
    {
        using var input = Activation.FromArray(mel, frames, mel.Length / frames);
        var x = TdnnRelu(_block0, input, par);
        Trace?.Invoke("spk_block0", x);
        int C = x.Channels;
        var outs = new List<Activation>();
        try
        {
            for (int bi = 0; bi < _blocks.Length; bi++)
            {
                var b = _blocks[bi];
                var prev = bi == 0 ? x : outs[^1];
                using var h = TdnnRelu(b.Tdnn1, prev, par);
                int w = C / Scale;
                using var res = new Activation(frames, C);
                using var part = new Activation(frames, w);
                Activation? last = null;
                for (int s = 0; s < Scale; s++)
                {
                    for (int t = 0; t < frames; t++)
                    {
                        var src = new ReadOnlySpan<float>(h.Row(t) + s * w, w);
                        var dst = new Span<float>(part.Row(t), w);
                        if (s >= 2) System.Numerics.Tensors.TensorPrimitives.Add(src, new ReadOnlySpan<float>(last!.Row(t), w), dst);
                        else src.CopyTo(dst);
                    }
                    Activation o = s == 0 ? part : TdnnRelu(b.Res2[s - 1], part, par);
                    for (int t = 0; t < frames; t++) new ReadOnlySpan<float>(o.Row(t), w).CopyTo(new Span<float>(res.Row(t) + s * w, w));
                    if (s > 0)
                    {
                        last?.Dispose();
                        last = o;
                    }
                }
                last?.Dispose();
                var y = TdnnRelu(b.Tdnn2, res, par);
                // Squeeze-excitation: channel gates from the time-mean.
                using var mean = new Activation(1, C, zero: true);
                for (int t = 0; t < frames; t++) System.Numerics.Tensors.TensorPrimitives.Add(mean.AsSpan(), new ReadOnlySpan<float>(y.Row(t), C), mean.AsSpan());
                System.Numerics.Tensors.TensorPrimitives.Multiply(mean.AsSpan(), 1f / frames, mean.AsSpan());
                using var s1 = b.Se1.Forward(mean, 0, 0, PadMode.Zero, par);
                Relu(s1.AsSpan());
                using var s2 = b.Se2.Forward(s1, 0, 0, PadMode.Zero, par);
                System.Numerics.Tensors.TensorPrimitives.Sigmoid(s2.AsSpan(), s2.AsSpan());
                for (int t = 0; t < frames; t++)
                {
                    var row = new Span<float>(y.Row(t), C);
                    System.Numerics.Tensors.TensorPrimitives.Multiply(row, s2.AsSpan(), row);
                    System.Numerics.Tensors.TensorPrimitives.Add(row, new ReadOnlySpan<float>(prev.Row(t), C), row);
                }
                outs.Add(y);
                Trace?.Invoke($"spk_block{bi + 1}", y);
            }

            // Multi-layer feature aggregation over blocks 1..3.
            using var cat = new Activation(frames, C * outs.Count);
            for (int t = 0; t < frames; t++)
                for (int i = 0; i < outs.Count; i++)
                    new ReadOnlySpan<float>(outs[i].Row(t), C).CopyTo(new Span<float>(cat.Row(t) + i * C, C));
            using var mfa = TdnnRelu(_mfa, cat, par);
            Trace?.Invoke("spk_mfa", mfa);

            var pooled = AttentiveStatsPool(mfa, par);
            using var pooledAct = Activation.FromArray(pooled, 1, pooled.Length);
            Trace?.Invoke("spk_asp", pooledAct);
            using var emb = _fc.Forward(pooledAct, 0, 0, PadMode.Zero, par);
            return emb.ToArray();
        }
        finally
        {
            x.Dispose();
            foreach (var o in outs) o.Dispose();
        }
    }

    private float[] AttentiveStatsPool(Activation x, Parallelism par)
    {
        int T = x.Time, C = x.Channels;
        var (mean, std) = Stats(x, null);
        using var attnIn = new Activation(T, 3 * C);
        for (int t = 0; t < T; t++)
        {
            new ReadOnlySpan<float>(x.Row(t), C).CopyTo(new Span<float>(attnIn.Row(t), C));
            mean.CopyTo(new Span<float>(attnIn.Row(t) + C, C));
            std.CopyTo(new Span<float>(attnIn.Row(t) + 2 * C, C));
        }
        using var a = TdnnRelu(_aspTdnn, attnIn, par);
        System.Numerics.Tensors.TensorPrimitives.Tanh(a.AsSpan(), a.AsSpan());
        using var logits = _aspConv.ForwardSame(a, PadMode.Reflect, par);
        // Softmax over time per channel.
        var w = new float[T * C];
        for (int c = 0; c < C; c++)
        {
            float max = float.NegativeInfinity;
            for (int t = 0; t < T; t++) max = MathF.Max(max, logits.Row(t)[c]);
            double sum = 0;
            for (int t = 0; t < T; t++)
            {
                float e = MathF.Exp(logits.Row(t)[c] - max);
                w[t * C + c] = e;
                sum += e;
            }
            float inv = (float)(1.0 / sum);
            for (int t = 0; t < T; t++) w[t * C + c] *= inv;
        }
        var (m2, s2) = Stats(x, w);
        return [.. m2, .. s2];
    }

    /// <summary>Weighted mean and std over time (uniform weights 1/T when <paramref name="w"/> is null),
    /// <c>std = sqrt(max(Σ w (x − μ)², 1e-12))</c>.</summary>
    private static (float[] Mean, float[] Std) Stats(Activation x, float[]? w)
    {
        int T = x.Time, C = x.Channels;
        var mean = new float[C];
        var std = new float[C];
        float uni = 1f / T;
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++) mean[c] += (w == null ? uni : w[t * C + c]) * x.Row(t)[c];
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
            {
                float d = x.Row(t)[c] - mean[c];
                std[c] += (w == null ? uni : w[t * C + c]) * d * d;
            }
        for (int c = 0; c < C; c++) std[c] = MathF.Sqrt(MathF.Max(std[c], 1e-12f));
        return (mean, std);
    }

    public void Dispose()
    {
        _block0.Dispose();
        foreach (var b in _blocks)
        {
            b.Tdnn1.Dispose();
            b.Tdnn2.Dispose();
            b.Se1.Dispose();
            b.Se2.Dispose();
            foreach (var r in b.Res2) r.Dispose();
        }
        _mfa.Dispose();
        _aspTdnn.Dispose();
        _aspConv.Dispose();
        _fc.Dispose();
    }
}
