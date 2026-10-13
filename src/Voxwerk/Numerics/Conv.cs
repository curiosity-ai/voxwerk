using Voxwerk.IO;

namespace Voxwerk.Numerics;

public enum PadMode { Zero, Replicate, Reflect }

/// <summary>SnakeBeta activation <c>x + sin²(αx) / (β + 1e-9)</c> with <c>exp(α_raw)</c> and
/// <c>1 / (exp(β_raw) + 1e-9)</c> precomputed per channel.</summary>
public sealed unsafe class Snake
{
    private readonly float[] _alpha, _invBeta;

    public Snake(float[] alphaRaw, float[] betaRaw)
    {
        _alpha = alphaRaw.Select(MathF.Exp).ToArray();
        _invBeta = betaRaw.Select(v => 1f / (MathF.Exp(v) + 1e-9f)).ToArray();
    }

    public static Snake Load(SafeTensorsFile st, string prefix) => new(st.ReadFloat(prefix + ".alpha"), st.ReadFloat(prefix + ".beta"));

    public int Channels => _alpha.Length;

    /// <summary>y = snake(x) for <paramref name="rows"/> channels-last rows (in place allowed).</summary>
    public void Apply(float* x, float* y, long rows, Parallelism par)
    {
        int C = _alpha.Length;
        fixed (float* a = _alpha, ib = _invBeta)
        {
            float* ap = a, ibp = ib;
            par.ForRange((int)rows, 512, (r0, r1) => Ops.Snake(x + (long)r0 * C, y + (long)r0 * C, r1 - r0, C, ap, ibp));
        }
    }

    public void ApplyInPlace(Activation x, Parallelism par) => Apply(x.Ptr, x.Ptr, x.Time, par);
}

internal static unsafe class ConvInput
{
    /// <summary>Returns the GEMM input for a convolution: <paramref name="x"/> itself when no padding or
    /// activation is needed, otherwise a pooled copy with <paramref name="pre"/> applied and the requested padding.</summary>
    public static Activation Prepare(Activation x, int left, int right, PadMode mode, Snake? pre, Parallelism par, out bool owned)
    {
        if (left == 0 && right == 0 && pre == null)
        {
            owned = false;
            return x;
        }
        owned = true;
        int C = x.Channels, T = x.Time;
        var p = new Activation(T + left + right, C);
        if (pre != null) pre.Apply(x.Ptr, p.Row(left), T, par);
        else Buffer.MemoryCopy(x.Ptr, p.Row(left), x.Length * 4, x.Length * 4);
        long rowBytes = (long)C * 4;
        // Padding rows replicate/reflect the (already activated) body, or are zero.
        for (int i = 0; i < left; i++)
        {
            int src = mode switch { PadMode.Replicate => 0, PadMode.Reflect => left - i, _ => -1 };
            if (src < 0 || T == 0) new Span<float>(p.Row(i), C).Clear();
            else Buffer.MemoryCopy(p.Row(left + Math.Min(src, T - 1)), p.Row(i), rowBytes, rowBytes);
        }
        for (int i = 0; i < right; i++)
        {
            int src = mode switch { PadMode.Replicate => T - 1, PadMode.Reflect => T - 2 - i, _ => -1 };
            if (src < 0) new Span<float>(p.Row(left + T + i), C).Clear();
            else Buffer.MemoryCopy(p.Row(left + src), p.Row(left + T + i), rowBytes, rowBytes);
        }
        return p;
    }
}

/// <summary>
/// 1-D convolution over channels-last input, lowered to one GEMM against a pre-packed weight:
/// output row t is <c>Σ_tap Xp[t·s + tap·d] · W[:, :, tap]ᵀ</c>. With dilation 1 the im2col row of t is a
/// contiguous slice of the padded input, so A rows simply overlap (no copy); dilated convolutions use a
/// tiled im2col buffer. Single-output-channel convolutions (the vocoder's last layer) use a direct dot.
/// </summary>
public sealed unsafe class Conv1d : IDisposable
{
    private readonly PackedMatrix? _w;
    private readonly float[]? _wFlat;   // [taps * cin] for cout == 1
    private readonly float[]? _bias;

    public Conv1d(float[] weight, float[]? bias, int cout, int cin, int taps, int stride = 1, int dilation = 1)
    {
        if (weight.Length != cout * cin * taps) throw new ArgumentException("Conv weight size mismatch.");
        Cout = cout;
        Cin = cin;
        Taps = taps;
        Stride = stride;
        Dilation = dilation;
        _bias = bias;
        if (cout == 1)
        {
            _wFlat = new float[taps * cin];
            for (int ci = 0; ci < cin; ci++)
                for (int t = 0; t < taps; t++) _wFlat[t * cin + ci] = weight[ci * taps + t];
        }
        else
        {
            fixed (float* w = weight) _w = PackedMatrix.FromConv(w, cout, cin, taps);
        }
    }

    public static Conv1d Load(SafeTensorsFile st, string prefix, int stride = 1, int dilation = 1, bool bias = true)
    {
        var info = st.Info(prefix + ".weight");
        int cout = info.Shape[0], cin = info.Shape[1], taps = info.Shape.Length > 2 ? info.Shape[2] : 1;
        return new Conv1d(st.ReadFloat(prefix + ".weight"), bias && st.Contains(prefix + ".bias") ? st.ReadFloat(prefix + ".bias") : null,
            cout, cin, taps, stride, dilation);
    }

    public int Cout { get; }
    public int Cin { get; }
    public int Taps { get; }
    public int Stride { get; }
    public int Dilation { get; }
    public int EffectiveKernel => (Taps - 1) * Dilation + 1;

    /// <summary>Convolves <paramref name="x"/> padded by (<paramref name="padLeft"/>, <paramref name="padRight"/>),
    /// optionally applying <paramref name="pre"/> to the input first, and either returns a new activation or
    /// adds the result into <paramref name="accumulateInto"/> (a fused residual connection; returns it).</summary>
    public Activation Forward(Activation x, int padLeft, int padRight, PadMode mode, Parallelism par, Snake? pre = null, Activation? accumulateInto = null)
    {
        if (x.Channels != Cin) throw new ArgumentException($"Expected {Cin} channels, got {x.Channels}.");
        int tp = x.Time + padLeft + padRight;
        int tout = (tp - EffectiveKernel) / Stride + 1;
        if (accumulateInto != null && (accumulateInto.Time != tout || accumulateInto.Channels != Cout))
            throw new ArgumentException("Residual target has the wrong shape.");
        var y = accumulateInto ?? new Activation(Math.Max(0, tout), Cout);
        if (tout <= 0) return y;
        var xp = ConvInput.Prepare(x, padLeft, padRight, mode, pre, par, out bool owned);
        try
        {
            bool acc = accumulateInto != null;
            fixed (float* bias = _bias)
            {
                if (_wFlat != null)
                {
                    if (Dilation != 1) throw new NotSupportedException("Dilated single-channel conv.");
                    fixed (float* w = _wFlat)
                    {
                        float* wp = w, xs = xp.Ptr, yd = y.Ptr;
                        float b = bias != null ? bias[0] : 0f;
                        long ld = (long)Stride * Cin;
                        int k = Taps * Cin;
                        par.ForRange(tout, 4096, (t0, t1) =>
                        {
                            var ws = new ReadOnlySpan<float>(wp, k);
                            for (int t = t0; t < t1; t++)
                            {
                                float v = System.Numerics.Tensors.TensorPrimitives.Dot(new ReadOnlySpan<float>(xs + t * ld, k), ws) + b;
                                yd[t] = acc ? yd[t] + v : v;
                            }
                        });
                    }
                }
                else
                {
                    Gemm.Conv(xp.Ptr, (long)Stride * Cin, (long)Dilation * Cin, Taps, Cin, tout, _w!, y.Ptr, Cout, par, bias, acc);
                }
            }
        }
        finally
        {
            if (owned) xp.Dispose();
        }
        return y;
    }

    /// <summary>Left context (input rows) a causal stride-1 application of this conv needs.</summary>
    public int History => EffectiveKernel - Stride;

    /// <summary>
    /// Streaming causal convolution (stride 1): convolves the next <paramref name="segment"/> given the previous
    /// <see cref="History"/> rows of (activated) input in <paramref name="state"/>, which is updated. Starting from a
    /// zeroed state, segment-by-segment results are bit-identical to <see cref="ForwardCausal"/> on the whole sequence.
    /// </summary>
    public Activation ForwardStreaming(Activation segment, Activation state, Parallelism par, Snake? pre = null, Activation? accumulateInto = null)
    {
        if (Stride != 1) throw new NotSupportedException("Streaming requires stride 1.");
        int hist = History, T = segment.Time;
        if (hist == 0) return Forward(segment, 0, 0, PadMode.Zero, par, pre, accumulateInto);
        using var xp = new Activation(hist + T, Cin);
        Buffer.MemoryCopy(state.Ptr, xp.Ptr, (long)hist * Cin * 4, (long)hist * Cin * 4);
        if (pre != null) pre.Apply(segment.Ptr, xp.Row(hist), T, par);
        else Buffer.MemoryCopy(segment.Ptr, xp.Row(hist), segment.Length * 4, segment.Length * 4);
        var y = Forward(xp, 0, 0, PadMode.Zero, par, null, accumulateInto);
        Buffer.MemoryCopy(xp.Row(T), state.Ptr, (long)hist * Cin * 4, (long)hist * Cin * 4);
        return y;
    }

    /// <summary>Causal convolution as in Mimi / Qwen3-TTS codecs: left pad <c>k_eff − stride</c>, plus the right
    /// padding needed for the last window to be complete.</summary>
    public Activation ForwardCausal(Activation x, PadMode mode, Parallelism par, Snake? pre = null, Activation? accumulateInto = null)
    {
        int padTotal = EffectiveKernel - Stride;
        double frames = (x.Time - EffectiveKernel + padTotal) / (double)Stride + 1;
        int ideal = ((int)Math.Ceiling(frames) - 1) * Stride + (EffectiveKernel - padTotal);
        int extra = ideal - x.Time;
        return Forward(x, padTotal, extra, mode, par, pre, accumulateInto);
    }

    /// <summary>PyTorch <c>padding="same"</c> (odd effective kernel, symmetric padding).</summary>
    public Activation ForwardSame(Activation x, PadMode mode, Parallelism par)
    {
        int total = Dilation * (Taps - 1);
        return Forward(x, total / 2, total - total / 2, mode, par);
    }

    public void Dispose() => _w?.Dispose();
}

/// <summary>
/// ConvTranspose1d (channels-last) with the causal right trim of <c>kernel − stride</c>, for kernels that are a
/// multiple m of the stride s. Output row <c>t·s + j</c> only sees inputs <c>t − m + 1 … t</c>, so the layer is a
/// single GEMM over overlapping input rows (depth m·Cin) producing s·Cout values per input row, which is exactly
/// the output laid out as [T·s, Cout] - no overlap-add pass and no zero-fill.
/// </summary>
public sealed unsafe class ConvTranspose1d : IDisposable
{
    private readonly PackedMatrix _w;
    private readonly float[]? _bias;   // replicated s times
    private readonly int _m;

    public ConvTranspose1d(SafeTensorsFile st, string prefix, int stride)
    {
        var info = st.Info(prefix + ".weight");
        Cin = info.Shape[0];
        Cout = info.Shape[1];
        Taps = info.Shape[2];
        Stride = stride;
        if (Taps % stride != 0) throw new NotSupportedException("ConvTranspose1d kernel must be a multiple of its stride.");
        _m = Taps / stride;
        var w = st.ReadFloat(prefix + ".weight");
        // B[(q · Cin + ci), j · Cout + co] = W[ci, co, j + (m − 1 − q) · s]
        int K = _m * Cin, N = Stride * Cout;
        var b = new float[(long)K * N];
        for (int q = 0; q < _m; q++)
            for (int ci = 0; ci < Cin; ci++)
                for (int j = 0; j < Stride; j++)
                    for (int co = 0; co < Cout; co++)
                        b[(long)(q * Cin + ci) * N + j * Cout + co] = w[((long)ci * Cout + co) * Taps + j + (_m - 1 - q) * Stride];
        fixed (float* bp = b) _w = PackedMatrix.FromKN(bp, K, N);
        if (st.Contains(prefix + ".bias"))
        {
            var bias = st.ReadFloat(prefix + ".bias");
            _bias = new float[N];
            for (int j = 0; j < Stride; j++) bias.CopyTo(_bias, j * Cout);
        }
    }

    public int Cin { get; }
    public int Cout { get; }
    public int Taps { get; }
    public int Stride { get; }

    /// <summary>Returns <c>T · stride</c> rows, optionally applying <paramref name="pre"/> to the input first.</summary>
    public Activation Forward(Activation x, Parallelism par, Snake? pre = null)
    {
        int T = x.Time;
        var y = new Activation(T * Stride, Cout);
        var xp = ConvInput.Prepare(x, _m - 1, 0, PadMode.Zero, pre, par, out bool owned);
        try
        {
            fixed (float* bias = _bias) Gemm.Conv(xp.Ptr, Cin, Cin, _m, Cin, T, _w, y.Ptr, (long)Stride * Cout, par, bias);
        }
        finally
        {
            if (owned) xp.Dispose();
        }
        return y;
    }

    /// <summary>Input rows of history a streaming application needs (kernel / stride − 1).</summary>
    public int History => _m - 1;

    /// <summary>Streaming counterpart of <see cref="Forward"/>; <paramref name="state"/> holds the previous
    /// <see cref="History"/> activated input rows (zero at the start) and is updated.</summary>
    public Activation ForwardStreaming(Activation segment, Activation state, Parallelism par, Snake? pre = null)
    {
        int hist = History, T = segment.Time;
        using var xp = new Activation(hist + T, Cin);
        if (hist > 0) Buffer.MemoryCopy(state.Ptr, xp.Ptr, (long)hist * Cin * 4, (long)hist * Cin * 4);
        if (pre != null) pre.Apply(segment.Ptr, xp.Row(hist), T, par);
        else Buffer.MemoryCopy(segment.Ptr, xp.Row(hist), segment.Length * 4, segment.Length * 4);
        var y = new Activation(T * Stride, Cout);
        fixed (float* bias = _bias) Gemm.Conv(xp.Ptr, Cin, Cin, _m, Cin, T, _w, y.Ptr, (long)Stride * Cout, par, bias);
        if (hist > 0) Buffer.MemoryCopy(xp.Row(T), state.Ptr, (long)hist * Cin * 4, (long)hist * Cin * 4);
        return y;
    }

    public void Dispose() => _w.Dispose();
}

/// <summary>A dense layer over channels-last rows with a pre-packed float32 weight.</summary>
public sealed unsafe class Linear : IDisposable
{
    private readonly PackedMatrix _w;
    private readonly float[]? _bias;

    public Linear(float[] weight, float[]? bias, int outDim, int inDim)
    {
        In = inDim;
        Out = outDim;
        fixed (float* w = weight) _w = PackedMatrix.FromLinear(w, outDim, inDim);
        _bias = bias;
    }

    /// <summary>Loads (and optionally row-concatenates) linear / 1×1-conv weights.</summary>
    public static Linear Load(SafeTensorsFile st, params string[] prefixes)
    {
        var ws = new List<float[]>();
        var bs = new List<float[]>();
        int inDim = 0, outDim = 0;
        foreach (var p in prefixes)
        {
            var info = st.Info(p + ".weight");
            inDim = info.Shape[1];
            outDim += info.Shape[0];
            ws.Add(st.ReadFloat(p + ".weight"));
            if (st.Contains(p + ".bias")) bs.Add(st.ReadFloat(p + ".bias"));
        }
        if (bs.Count != 0 && bs.Count != prefixes.Length) throw new InvalidDataException("Mixed bias presence.");
        return new Linear(ws.SelectMany(w => w).ToArray(), bs.Count == 0 ? null : bs.SelectMany(b => b).ToArray(), outDim, inDim);
    }

    public int In { get; }
    public int Out { get; }

    public void Forward(float* x, long ldx, int rows, float* y, long ldy, Parallelism par, bool accumulate = false)
    {
        fixed (float* b = _bias) Gemm.Multiply(x, ldx, rows, _w, y, ldy, par, accumulate ? null : b, accumulate);
    }

    public Activation Forward(Activation x, Parallelism par)
    {
        var y = new Activation(x.Time, Out);
        Forward(x.Ptr, In, x.Time, y.Ptr, Out, par);
        return y;
    }

    public void Dispose() => _w.Dispose();
}
