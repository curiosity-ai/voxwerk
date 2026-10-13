using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Voxwerk.Numerics;

/// <summary>Causal scaled-dot-product attention with grouped KV heads and an optional sliding window.</summary>
public static unsafe class Attention
{
    /// <summary>
    /// For query row t (absolute position <c>pos0 + t</c>) attends over cached keys
    /// <c>j ∈ [max(0, p − window + 1), p]</c> (<paramref name="window"/> ≤ 0 means unbounded).
    /// Rows are laid out <c>[position][head · headDim]</c>.
    /// </summary>
    public static void Causal(float* q, int qStride, int queries, int pos0,
                              float* k, float* v, int kvStride,
                              int heads, int kvHeads, int headDim, int window,
                              float* output, int outStride, Parallelism par)
    {
        float scale = 1f / MathF.Sqrt(headDim);
        int group = heads / kvHeads;
        int work = heads * queries;
        int maxKeys = pos0 + queries;
        par.ForRange(work, 4, (w0, w1) =>
        {
            var scores = new float[maxKeys];
            for (int w = w0; w < w1; w++)
            {
                int h = w % heads, t = w / heads;
                int p = pos0 + t;
                int lo = window > 0 ? Math.Max(0, p - window + 1) : 0;
                int n = p - lo + 1;
                var qs = new ReadOnlySpan<float>(q + (long)t * qStride + h * headDim, headDim);
                int kvh = h / group;
                for (int j = 0; j < n; j++)
                    scores[j] = TensorPrimitives.Dot(qs, new ReadOnlySpan<float>(k + (long)(lo + j) * kvStride + kvh * headDim, headDim)) * scale;
                var sc = new Span<float>(scores, 0, n);
                Ops.Softmax(sc);
                var o = new Span<float>(output + (long)t * outStride + h * headDim, headDim);
                o.Clear();
                for (int j = 0; j < n; j++)
                    TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(v + (long)(lo + j) * kvStride + kvh * headDim, headDim), scores[j], o, o);
            }
        });
    }

    /// <summary>
    /// Causal attention against a head-major KV cache (<c>k[kvHead][capacity][headDim]</c>), as used for
    /// autoregressive decoding: each KV head's keys/values are streamed once, contiguously, for all the
    /// query heads that share it (GQA), which keeps long-context decode steps bandwidth-efficient.
    /// Query rows are <c>[t][heads · headDim]</c> with absolute position <c>pos0 + t</c>.
    /// </summary>
    public static void CausalCached(float* q, int qStride, int queries, int pos0, float* k, float* v, int capacity,
                                    int heads, int kvHeads, int headDim, float* output, int outStride, Parallelism par)
    {
        float scale = 1f / MathF.Sqrt(headDim);
        int group = heads / kvHeads;
        int keysMax = pos0 + queries;
        par.ForRange(kvHeads * queries, 1, (w0, w1) =>
        {
            var scores = new float[group * keysMax];
            for (int w = w0; w < w1; w++)
            {
                int h = w % kvHeads, t = w / kvHeads;
                int n = pos0 + t + 1;
                float* kh = k + (long)h * capacity * headDim;
                float* vh = v + (long)h * capacity * headDim;
                fixed (float* sc = scores)
                {
                    for (int g = 0; g < group; g++)
                    {
                        float* qg = q + (long)t * qStride + (h * group + g) * headDim;
                        float* sg = sc + g * keysMax;
                        for (int j = 0; j < n; j++) sg[j] = Dot(qg, kh + (long)j * headDim, headDim) * scale;
                        Ops.Softmax(new Span<float>(sg, n));
                    }
                    for (int g = 0; g < group; g++)
                    {
                        float* o = output + (long)t * outStride + (h * group + g) * headDim;
                        new Span<float>(o, headDim).Clear();
                        float* sg = sc + g * keysMax;
                        for (int j = 0; j < n; j++) Axpy(sg[j], vh + (long)j * headDim, o, headDim);
                    }
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Dot(float* a, float* b, int n)
    {
        if (Vector512.IsHardwareAccelerated && (n & 31) == 0)
        {
            var acc0 = Vector512<float>.Zero;
            var acc1 = Vector512<float>.Zero;
            for (int i = 0; i < n; i += 32)
            {
                acc0 = Vector512.FusedMultiplyAdd(Vector512.Load(a + i), Vector512.Load(b + i), acc0);
                acc1 = Vector512.FusedMultiplyAdd(Vector512.Load(a + i + 16), Vector512.Load(b + i + 16), acc1);
            }
            return Vector512.Sum(acc0 + acc1);
        }
        return TensorPrimitives.Dot(new ReadOnlySpan<float>(a, n), new ReadOnlySpan<float>(b, n));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Axpy(float alpha, float* x, float* y, int n)
    {
        if (Vector512.IsHardwareAccelerated && (n & 15) == 0)
        {
            var a = Vector512.Create(alpha);
            for (int i = 0; i < n; i += 16)
                Vector512.Store(Vector512.FusedMultiplyAdd(a, Vector512.Load(x + i), Vector512.Load(y + i)), y + i);
            return;
        }
        TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(x, n), alpha, new ReadOnlySpan<float>(y, n), new Span<float>(y, n));
    }
}
