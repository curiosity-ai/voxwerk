namespace Voxwerk.Numerics;

/// <summary>
/// The right-hand operand <c>B[K, N]</c> of a GEMM, pre-packed into column panels of
/// <see cref="NR"/> = 32 outputs: element <c>(k, n)</c> lives at <c>[(n / 32) * K + k] * 32 + n % 32</c>,
/// with the last panel zero-padded. A linear layer with PyTorch weight <c>W[out, in]</c> is
/// <c>B = Wᵀ</c> (K = in, N = out). The panel is the exact layout the GEMM micro-kernel and the
/// small-batch GEMV stream, so weights are reordered once at load time and never again.
/// Panels are stored either as float32 or as bfloat16 (the talker checkpoint's native precision).
/// </summary>
public sealed unsafe class PackedMatrix : IDisposable
{
    public const int NR = 32;

    private readonly NativeBuffer<float>? _f32;
    private readonly NativeBuffer<ushort>? _bf16;

    private PackedMatrix(int k, int n, NativeBuffer<float>? f32, NativeBuffer<ushort>? bf16)
    {
        K = k;
        N = n;
        _f32 = f32;
        _bf16 = bf16;
    }

    public int K { get; }
    public int N { get; }
    public int Blocks => (N + NR - 1) / NR;
    public bool IsBf16 => _bf16 != null;
    internal float* F32 => _f32!.Ptr;
    internal ushort* B16 => _bf16!.Ptr;

    public long ByteSize => _f32 != null ? _f32.Length * 4 : _bf16!.Length * 2;

    /// <summary>Packs a PyTorch linear weight <c>W[n, k]</c> (row-major, float32).</summary>
    public static PackedMatrix FromLinear(float* w, int n, int k)
    {
        var buf = new NativeBuffer<float>((long)((n + NR - 1) / NR) * k * NR);
        float* dst = buf.Ptr;
        Parallel.For(0, (n + NR - 1) / NR, nb =>
        {
            float* d = dst + (long)nb * k * NR;
            for (int j = 0; j < NR && nb * NR + j < n; j++)
            {
                float* row = w + (long)(nb * NR + j) * k;
                for (int kk = 0; kk < k; kk++) d[(long)kk * NR + j] = row[kk];
            }
        });
        return new PackedMatrix(k, n, buf, null);
    }

    /// <summary>Packs a row-major float32 matrix <c>B[k, n]</c>.</summary>
    public static PackedMatrix FromKN(float* b, int k, int n)
    {
        var buf = new NativeBuffer<float>((long)((n + NR - 1) / NR) * k * NR);
        float* dst = buf.Ptr;
        Parallel.For(0, (n + NR - 1) / NR, nb =>
        {
            float* d = dst + (long)nb * k * NR;
            int cols = Math.Min(NR, n - nb * NR);
            for (int kk = 0; kk < k; kk++)
                for (int j = 0; j < cols; j++) d[(long)kk * NR + j] = b[(long)kk * n + nb * NR + j];
        });
        return new PackedMatrix(k, n, buf, null);
    }

    /// <summary>Packs a Conv1d weight <c>W[cout, cin, taps]</c> as <c>B[tap * cin + ci, co]</c>, the operand of
    /// a convolution over channels-last input whose im2col row for output t is the concatenation of the
    /// input rows touched by each tap.</summary>
    public static PackedMatrix FromConv(float* w, int cout, int cin, int taps)
    {
        int k = cin * taps;
        var buf = new NativeBuffer<float>((long)((cout + NR - 1) / NR) * k * NR);
        float* dst = buf.Ptr;
        Parallel.For(0, (cout + NR - 1) / NR, nb =>
        {
            float* d = dst + (long)nb * k * NR;
            for (int j = 0; j < NR && nb * NR + j < cout; j++)
            {
                float* src = w + (long)(nb * NR + j) * cin * taps;
                for (int ci = 0; ci < cin; ci++)
                    for (int t = 0; t < taps; t++)
                        d[((long)t * cin + ci) * NR + j] = src[ci * taps + t];
            }
        });
        return new PackedMatrix(k, cout, buf, null);
    }

    /// <summary>Packs one or more bf16 linear weights <c>W_i[n_i, k]</c> stacked along the output dimension
    /// (e.g. fused q/k/v projections) without changing their precision.</summary>
    public static PackedMatrix FromBf16Linear(int k, params NativeBuffer<ushort>[] parts)
    {
        int n = 0;
        foreach (var p in parts)
        {
            if (p.Length % k != 0) throw new ArgumentException("Weight size is not a multiple of the input dimension.");
            n += (int)(p.Length / k);
        }
        var rows = new (IntPtr Ptr, int Count)[parts.Length];
        for (int i = 0; i < parts.Length; i++) rows[i] = ((IntPtr)parts[i].Ptr, (int)(parts[i].Length / k));
        var buf = new NativeBuffer<ushort>((long)((n + NR - 1) / NR) * k * NR);
        ushort* dst = buf.Ptr;
        Parallel.For(0, (n + NR - 1) / NR, nb =>
        {
            ushort* d = dst + (long)nb * k * NR;
            for (int j = 0; j < NR && nb * NR + j < n; j++)
            {
                int col = nb * NR + j, part = 0;
                while (col >= rows[part].Count) col -= rows[part++].Count;
                ushort* row = (ushort*)rows[part].Ptr + (long)col * k;
                for (int kk = 0; kk < k; kk++) d[(long)kk * NR + j] = row[kk];
            }
        });
        return new PackedMatrix(k, n, null, buf);
    }

    public void Dispose()
    {
        _f32?.Dispose();
        _bf16?.Dispose();
    }
}
