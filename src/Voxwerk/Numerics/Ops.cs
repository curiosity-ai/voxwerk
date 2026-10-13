using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Voxwerk.Numerics;

/// <summary>Row-wise and element-wise kernels shared by the transformer and convolution stacks.</summary>
public static unsafe class Ops
{
    /// <summary>T5/Qwen RMSNorm over rows of <paramref name="dim"/>: <c>y = w · x / sqrt(mean(x²) + eps)</c>.</summary>
    public static void RmsNorm(float* x, float* y, int rows, int dim, float* weight, float eps)
    {
        for (int r = 0; r < rows; r++)
        {
            var xs = new ReadOnlySpan<float>(x + (long)r * dim, dim);
            var ys = new Span<float>(y + (long)r * dim, dim);
            float ss = TensorPrimitives.SumOfSquares(xs);
            float inv = 1f / MathF.Sqrt(ss / dim + eps);
            TensorPrimitives.Multiply(xs, inv, ys);
            TensorPrimitives.Multiply(ys, new ReadOnlySpan<float>(weight, dim), ys);
        }
    }

    /// <summary>LayerNorm over rows (biased variance, like torch).</summary>
    public static void LayerNorm(float* x, float* y, int rows, int dim, float* weight, float* bias, float eps)
    {
        for (int r = 0; r < rows; r++)
        {
            var xs = new ReadOnlySpan<float>(x + (long)r * dim, dim);
            var ys = new Span<float>(y + (long)r * dim, dim);
            float mean = TensorPrimitives.Sum(xs) / dim;
            TensorPrimitives.Subtract(xs, mean, ys);
            float var = TensorPrimitives.SumOfSquares(ys) / dim;
            float inv = 1f / MathF.Sqrt(var + eps);
            TensorPrimitives.Multiply(ys, inv, ys);
            TensorPrimitives.FusedMultiplyAdd(ys, new ReadOnlySpan<float>(weight, dim), new ReadOnlySpan<float>(bias, dim), ys);
        }
    }

    /// <summary>In-place SiLU-gated product: <c>g = silu(g) · u</c>.</summary>
    public static void SwiGlu(float* gate, float* up, long n)
    {
        for (long i = 0; i < n; i += 1 << 20)
        {
            int len = (int)Math.Min(1 << 20, n - i);
            var g = new Span<float>(gate + i, len);
            var u = new ReadOnlySpan<float>(up + i, len);
            // silu(x) = x * sigmoid(x)
            Span<float> tmp = len <= 8192 ? stackalloc float[len] : new float[len];
            TensorPrimitives.Sigmoid(g, tmp);
            TensorPrimitives.Multiply(g, tmp, g);
            TensorPrimitives.Multiply(g, u, g);
        }
    }

    public static void Silu(float* x, long n)
    {
        for (long i = 0; i < n; i += 4096)
        {
            int len = (int)Math.Min(4096, n - i);
            var g = new Span<float>(x + i, len);
            Span<float> tmp = stackalloc float[len];
            TensorPrimitives.Sigmoid(g, tmp);
            TensorPrimitives.Multiply(g, tmp, g);
        }
    }

    /// <summary>Exact (erf-based) GELU, torch's default <c>nn.GELU()</c> / <c>ACT2FN["gelu"]</c>.</summary>
    public static void Gelu(float* x, long n)
    {
        long i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            var half = Vector512.Create(0.5f);
            var rs2 = Vector512.Create(0.70710678118654752f);
            for (; i <= n - 16; i += 16)
            {
                var v = Vector512.Load(x + i);
                Vector512.Store(half * v * (Vector512<float>.One + Erf(v * rs2)), x + i);
            }
        }
        for (; i < n; i++) x[i] = 0.5f * x[i] * (1f + ErfScalar(x[i] * 0.70710678118654752f));
    }

    /// <summary>ELU with alpha = 1.</summary>
    public static void Elu(float* x, float* y, long n)
    {
        long i = 0;
        if (Vector512.IsHardwareAccelerated)
        {
            for (; i <= n - 16; i += 16)
            {
                var v = Vector512.Load(x + i);
                var neg = Exp(Vector512.Min(v, Vector512<float>.Zero)) - Vector512<float>.One;
                Vector512.Store(Vector512.ConditionalSelect(Vector512.GreaterThan(v, Vector512<float>.Zero), v, neg), y + i);
            }
        }
        for (; i < n; i++) y[i] = x[i] > 0 ? x[i] : MathF.Exp(x[i]) - 1f;
    }

    /// <summary>
    /// SnakeBeta over channels-last rows: <c>y = x + 1/(β + 1e-9) · sin²(α · x)</c> with per-channel
    /// <paramref name="alpha"/> = exp(α_raw) and <paramref name="invBeta"/> = 1/(exp(β_raw) + 1e-9) precomputed.
    /// </summary>
    public static void Snake(float* x, float* y, long rows, int channels, float* alpha, float* invBeta)
    {
        for (long r = 0; r < rows; r++)
        {
            float* xr = x + r * channels;
            float* yr = y + r * channels;
            int c = 0;
            if (Vector512.IsHardwareAccelerated)
            {
                for (; c <= channels - 16; c += 16)
                {
                    var v = Vector512.Load(xr + c);
                    var s = Sin(v * Vector512.Load(alpha + c));
                    Vector512.Store(Vector512.FusedMultiplyAdd(s * s, Vector512.Load(invBeta + c), v), yr + c);
                }
            }
            for (; c < channels; c++)
            {
                float s = MathF.Sin(xr[c] * alpha[c]);
                yr[c] = xr[c] + invBeta[c] * (s * s);
            }
        }
    }

    /// <summary>Numerically stable softmax in place.</summary>
    public static void Softmax(Span<float> x)
    {
        float max = TensorPrimitives.Max(x);
        TensorPrimitives.Subtract(x, max, x);
        TensorPrimitives.Exp(x, x);
        float sum = TensorPrimitives.Sum(x);
        TensorPrimitives.Multiply(x, 1f / sum, x);
    }


    /// <summary>Vectorized sin (Cephes sinf: Cody-Waite reduction by π/2, degree-7/8 polynomials). Max abs error
    /// ≈ 1e-7 for |x| ≤ 1e4. <c>Vector512.Sin</c> in .NET 10 is ~5× slower, which matters at audio rate.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Sin(Vector512<float> x)
    {
        var j = Vector512.Round(x * 0.636619772367581343f);
        var r = Vector512.FusedMultiplyAdd(j, Vector512.Create(-1.5703125f), x);
        r = Vector512.FusedMultiplyAdd(j, Vector512.Create(-4.837512969970703125e-4f), r);
        r = Vector512.FusedMultiplyAdd(j, Vector512.Create(-7.54978995489188216e-8f), r);
        var q = Vector512.ConvertToInt32(j);
        var r2 = r * r;
        var s = Vector512.FusedMultiplyAdd(Vector512.Create(-1.9515295891e-4f), r2, Vector512.Create(8.3321608736e-3f));
        s = Vector512.FusedMultiplyAdd(s, r2, Vector512.Create(-1.6666654611e-1f));
        s = Vector512.FusedMultiplyAdd(s * r2, r, r);
        var c = Vector512.FusedMultiplyAdd(Vector512.Create(2.443315711809948e-5f), r2, Vector512.Create(-1.388731625493765e-3f));
        c = Vector512.FusedMultiplyAdd(c, r2, Vector512.Create(4.166664568298827e-2f));
        c = Vector512.FusedMultiplyAdd(c * r2, r2, Vector512.FusedMultiplyAdd(Vector512.Create(-0.5f), r2, Vector512<float>.One));
        var useCos = Vector512.Equals(q & Vector512.Create(1), Vector512.Create(1)).AsSingle();
        var res = Vector512.ConditionalSelect(useCos, c, s);
        var neg = Vector512.Equals(q & Vector512.Create(2), Vector512.Create(2)).AsSingle();
        return Vector512.ConditionalSelect(neg, -res, res);
    }

    /// <summary>Vectorized exp (Cephes expf, ~1 ulp). Inputs are clamped to [−87.3, 88.7].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Exp(Vector512<float> x)
    {
        x = Vector512.Min(Vector512.Max(x, Vector512.Create(-87.3f)), Vector512.Create(88.7f));
        var fx = Vector512.Round(x * 1.44269504088896341f);
        var r = Vector512.FusedMultiplyAdd(fx, Vector512.Create(-0.693359375f), x);
        r = Vector512.FusedMultiplyAdd(fx, Vector512.Create(2.12194440e-4f), r);
        var y = Vector512.FusedMultiplyAdd(Vector512.Create(1.9875691500e-4f), r, Vector512.Create(1.3981999507e-3f));
        y = Vector512.FusedMultiplyAdd(y, r, Vector512.Create(8.3334519073e-3f));
        y = Vector512.FusedMultiplyAdd(y, r, Vector512.Create(4.1665795894e-2f));
        y = Vector512.FusedMultiplyAdd(y, r, Vector512.Create(1.6666665459e-1f));
        y = Vector512.FusedMultiplyAdd(y, r, Vector512.Create(5.0000001201e-1f));
        y = Vector512.FusedMultiplyAdd(y, r * r, r + Vector512<float>.One);
        var pow2 = Vector512.ShiftLeft(Vector512.ConvertToInt32(fx) + Vector512.Create(127), 23).AsSingle();
        return y * pow2;
    }

    // erf via Abramowitz & Stegun 7.1.26 (max abs error 1.5e-7) - the same polynomial PyTorch's vectorized
    // CPU erf uses, so GELU tracks the reference closely.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector512<float> Erf(Vector512<float> x)
    {
        var a = Vector512.Abs(x);
        var t = Vector512<float>.One / Vector512.FusedMultiplyAdd(Vector512.Create(0.3275911f), a, Vector512<float>.One);
        var p = Vector512.Create(1.061405429f);
        p = Vector512.FusedMultiplyAdd(p, t, Vector512.Create(-1.453152027f));
        p = Vector512.FusedMultiplyAdd(p, t, Vector512.Create(1.421413741f));
        p = Vector512.FusedMultiplyAdd(p, t, Vector512.Create(-0.284496736f));
        p = Vector512.FusedMultiplyAdd(p, t, Vector512.Create(0.254829592f));
        p *= t;
        var r = Vector512<float>.One - p * Exp(-(a * a));
        return Vector512.CopySign(r, x);
    }

    internal static float ErfScalar(float x)
    {
        float a = MathF.Abs(x);
        float t = 1f / (1f + 0.3275911f * a);
        float p = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
        return MathF.CopySign(1f - p * MathF.Exp(-a * a), x);
    }
}
