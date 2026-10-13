using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Voxwerk.Numerics;

/// <summary>
/// Single-precision GEMM <c>C[m, N] (+)= A[m, K] · B[K, N] (+ bias)</c> with B a <see cref="PackedMatrix"/>.
/// <para>
/// Goto-style blocking: A is packed per (MC × KC) block into MR-row micro-panels (L2 resident), the
/// B panel of 32 columns × KC (16 KB, widened from bf16 when needed) stays in L1, and an MR × 32 register
/// tile of FMA accumulators sweeps it - 12 × 32 on AVX-512 (24 of the 32 zmm registers), 2 × 32 on AVX2.
/// Each output is a single FMA chain over k, like oneDNN/MKL's sgemm.
/// </para>
/// <para>
/// For m ≤ 4 (autoregressive decoding) the product is memory-bound, so a separate path streams each
/// weight panel exactly once from memory and broadcasts the activations against it, which is what makes
/// keeping the talker's weights in bf16 pay off: half the bytes per token of a float32 run.
/// </para>
/// </summary>
public static unsafe class Gemm
{
    private const int InitZero = 0, InitLoad = 1, InitBias = 2, InitLoadBias = 3;
    private const int NR = PackedMatrix.NR;
    private const int KC = 256;          // depth of one packed block: A micro-panel 12 × 384 (18 KB) stays in L1
    private const int NC = 16 * NR;      // B block 384 × 512 (768 KB) streams from L2

    private static readonly bool UseAvx512 = Avx512F.IsSupported;
    private static readonly bool UseAvx2 = !UseAvx512 && Avx2.IsSupported && Fma.IsSupported;
    internal static readonly int MR = UseAvx512 ? 12 : UseAvx2 ? 2 : 1;
    private static readonly int MC = MR * 20;

    [ThreadStatic] private static Scratch? t_scratch;

    private sealed class Scratch
    {
        public readonly NativeBuffer<float> APack = new((long)MC * KC);
        public readonly NativeBuffer<float> BBlock = new((long)KC * NC);
        public readonly NativeBuffer<float> CTile = new((long)MR * NR);
        public readonly NativeBuffer<float> Bias = new(NC);
    }

    /// <summary>
    /// <c>C = A·B</c> (or <c>C += A·B</c> when <paramref name="accumulate"/>), plus <paramref name="bias"/>[n] when
    /// given (with <paramref name="accumulate"/> this is <c>C += A·B + bias</c>, i.e. a fused residual add). A rows are <paramref name="lda"/> apart and may overlap,
    /// which lets a stride-s convolution read its im2col rows straight from the padded input.
    /// </summary>
    public static void Multiply(float* a, long lda, int m, PackedMatrix b, float* c, long ldc, Parallelism par,
                                float* bias = null, bool accumulate = false)
    {
        if (m <= 0) return;
        if (m <= 4)
        {
            SmallM(a, lda, m, b, c, ldc, par, bias, accumulate);
            return;
        }
        int blocks = b.Blocks;
        int mChunks = (m + MC - 1) / MC;
        int nGroups = 1;
        if (par.MaxThreads > 1 && mChunks < par.MaxThreads)
            nGroups = Math.Min(blocks, (par.MaxThreads + mChunks - 1) / mChunks);
        int items = mChunks * nGroups;
        par.For(items, item =>
        {
            int mc = item / nGroups, g = item % nGroups;
            int nb0 = (int)((long)g * blocks / nGroups), nb1 = (int)((long)(g + 1) * blocks / nGroups);
            int m0 = mc * MC, m1 = Math.Min(m, m0 + MC);
            Block(a, lda, m0, m1, b, nb0, nb1, c, ldc, bias, accumulate);
        });
    }

    private static void Block(float* a, long lda, int m0, int m1, PackedMatrix b, int nb0, int nb1, float* c, long ldc, float* bias, bool accumulate)
    {
        var s = t_scratch ??= new Scratch();
        float* aPack = s.APack.Ptr, bBlock = s.BBlock.Ptr, cTile = s.CTile.Ptr, biasTile = s.Bias.Ptr;
        int K = b.K, N = b.N;
        int rows = m1 - m0;
        int tiles = (rows + MR - 1) / MR;
        const int panelsPerBlock = NC / NR;
        for (int k0 = 0; k0 < K; k0 += KC)
        {
            int kc = Math.Min(KC, K - k0);
            PackA(a + (long)m0 * lda + k0, lda, rows, kc, aPack);
            bool first = k0 == 0;
            int mode = !first ? InitLoad : accumulate ? (bias != null ? InitLoadBias : InitLoad) : bias != null ? InitBias : InitZero;
            for (int nbBlock = nb0; nbBlock < nb1; nbBlock += panelsPerBlock)
            {
                int nbEnd = Math.Min(nb1, nbBlock + panelsPerBlock);
                bool directBf16 = b.IsBf16 && UseAvx512;   // the AVX-512 kernel widens bf16 panels in registers
                if (b.IsBf16 && !directBf16)
                {
                    for (int nb = nbBlock; nb < nbEnd; nb++)
                        Bf16.ToFloat(b.B16 + ((long)nb * K + k0) * NR, bBlock + (long)(nb - nbBlock) * kc * NR, kc * NR);
                }
                else if (b.IsInt8)
                {
                    for (int nb = nbBlock; nb < nbEnd; nb++)
                        DequantizeInt8(b.I8 + ((long)nb * K + k0) * NR, b.Scale + nb * NR, bBlock + (long)(nb - nbBlock) * kc * NR, kc);
                }
                if (mode == InitBias || mode == InitLoadBias)
                {
                    for (int j = 0; j < (nbEnd - nbBlock) * NR; j++)
                    {
                        int col = nbBlock * NR + j;
                        biasTile[j] = col < N ? bias[col] : 0f;
                    }
                }
                for (int t = 0; t < tiles; t++)
                {
                    int r0 = m0 + t * MR;
                    int tr = Math.Min(MR, m1 - r0);
                    float* ap = aPack + (long)t * kc * MR;
                    for (int nb = nbBlock; nb < nbEnd; nb++)
                    {
                        float* panel = (b.IsBf16 && !directBf16) || b.IsInt8 ? bBlock + (long)(nb - nbBlock) * kc * NR : directBf16 ? null : b.F32 + ((long)nb * K + k0) * NR;
                        ushort* panel16 = directBf16 ? b.B16 + ((long)nb * K + k0) * NR : null;
                        float* bp = biasTile + (nb - nbBlock) * NR;
                        int cols = Math.Min(NR, N - nb * NR);
                        float* cp = c + (long)r0 * ldc + nb * NR;
                        if (tr == MR && cols == NR)
                        {
                            if (panel16 != null) Kernel512Bf16(ap, panel16, kc, cp, ldc, mode, bp);
                            else Kernel(ap, panel, kc, cp, ldc, mode, bp);
                        }
                        else
                        {
                            if (mode == InitLoad || mode == InitLoadBias)
                            {
                                new Span<float>(cTile, MR * NR).Clear();
                                for (int i = 0; i < tr; i++)
                                    for (int j = 0; j < cols; j++) cTile[i * NR + j] = cp[i * ldc + j];
                            }
                            if (panel16 != null) Kernel512Bf16(ap, panel16, kc, cTile, NR, mode, bp);
                            else Kernel(ap, panel, kc, cTile, NR, mode, bp);
                            for (int i = 0; i < tr; i++)
                                for (int j = 0; j < cols; j++) cp[i * ldc + j] = cTile[i * NR + j];
                        }
                    }
                }
            }
        }
    }

    /// <summary>Packs <paramref name="rows"/> × <paramref name="kc"/> of A into MR-row micro-panels laid out
    /// [tile][k][MR] (zero-padded), so the kernel's broadcasts are sequential loads.</summary>
    private static void PackA(float* a, long lda, int rows, int kc, float* dst)
    {
        int tiles = (rows + MR - 1) / MR;
        for (int t = 0; t < tiles; t++)
        {
            float* d = dst + (long)t * kc * MR;
            int r0 = t * MR;
            int tr = Math.Min(MR, rows - r0);
            if (tr < MR) new Span<float>(d, kc * MR).Clear();
            for (int i = 0; i < tr; i++)
            {
                float* src = a + (r0 + i) * lda;
                float* di = d + i;
                int p = 0;
                for (; p <= kc - 4; p += 4)
                {
                    di[p * MR] = src[p];
                    di[(p + 1) * MR] = src[p + 1];
                    di[(p + 2) * MR] = src[p + 2];
                    di[(p + 3) * MR] = src[p + 3];
                }
                for (; p < kc; p++) di[p * MR] = src[p];
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Kernel(float* a, float* b, int kc, float* c, long ldc, int mode, float* bias)
    {
        if (UseAvx512) Kernel512(a, b, kc, c, ldc, mode, bias);
        else if (UseAvx2) Kernel256(a, b, kc, c, ldc, mode, bias);
        else KernelScalar(a, b, kc, c, ldc, mode, bias);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel512(float* a, float* b, int kc, float* c, long ldc, int mode, float* bias)
    {
        Vector512<float> c0a, c0b, c1a, c1b, c2a, c2b, c3a, c3b, c4a, c4b, c5a, c5b, c6a, c6b, c7a, c7b, c8a, c8b, c9a, c9b, c10a, c10b, c11a, c11b;
        if (mode >= InitLoad && mode != InitBias)
        {
            c0a = Vector512.Load(c + 0 * ldc); c0b = Vector512.Load(c + 0 * ldc + 16);
            c1a = Vector512.Load(c + 1 * ldc); c1b = Vector512.Load(c + 1 * ldc + 16);
            c2a = Vector512.Load(c + 2 * ldc); c2b = Vector512.Load(c + 2 * ldc + 16);
            c3a = Vector512.Load(c + 3 * ldc); c3b = Vector512.Load(c + 3 * ldc + 16);
            c4a = Vector512.Load(c + 4 * ldc); c4b = Vector512.Load(c + 4 * ldc + 16);
            c5a = Vector512.Load(c + 5 * ldc); c5b = Vector512.Load(c + 5 * ldc + 16);
            c6a = Vector512.Load(c + 6 * ldc); c6b = Vector512.Load(c + 6 * ldc + 16);
            c7a = Vector512.Load(c + 7 * ldc); c7b = Vector512.Load(c + 7 * ldc + 16);
            c8a = Vector512.Load(c + 8 * ldc); c8b = Vector512.Load(c + 8 * ldc + 16);
            c9a = Vector512.Load(c + 9 * ldc); c9b = Vector512.Load(c + 9 * ldc + 16);
            c10a = Vector512.Load(c + 10 * ldc); c10b = Vector512.Load(c + 10 * ldc + 16);
            c11a = Vector512.Load(c + 11 * ldc); c11b = Vector512.Load(c + 11 * ldc + 16);
            if (mode == InitLoadBias)
            {
                var ba = Vector512.Load(bias); var bb = Vector512.Load(bias + 16);
                c0a += ba; c0b += bb;
                c1a += ba; c1b += bb;
                c2a += ba; c2b += bb;
                c3a += ba; c3b += bb;
                c4a += ba; c4b += bb;
                c5a += ba; c5b += bb;
                c6a += ba; c6b += bb;
                c7a += ba; c7b += bb;
                c8a += ba; c8b += bb;
                c9a += ba; c9b += bb;
                c10a += ba; c10b += bb;
                c11a += ba; c11b += bb;
            }
        }
        else if (mode == InitBias)
        {
            var ba = Vector512.Load(bias); var bb = Vector512.Load(bias + 16);
            c0a = ba; c0b = bb;
            c1a = ba; c1b = bb;
            c2a = ba; c2b = bb;
            c3a = ba; c3b = bb;
            c4a = ba; c4b = bb;
            c5a = ba; c5b = bb;
            c6a = ba; c6b = bb;
            c7a = ba; c7b = bb;
            c8a = ba; c8b = bb;
            c9a = ba; c9b = bb;
            c10a = ba; c10b = bb;
            c11a = ba; c11b = bb;
        }
        else
        {
            c0a = Vector512<float>.Zero; c0b = Vector512<float>.Zero;
            c1a = Vector512<float>.Zero; c1b = Vector512<float>.Zero;
            c2a = Vector512<float>.Zero; c2b = Vector512<float>.Zero;
            c3a = Vector512<float>.Zero; c3b = Vector512<float>.Zero;
            c4a = Vector512<float>.Zero; c4b = Vector512<float>.Zero;
            c5a = Vector512<float>.Zero; c5b = Vector512<float>.Zero;
            c6a = Vector512<float>.Zero; c6b = Vector512<float>.Zero;
            c7a = Vector512<float>.Zero; c7b = Vector512<float>.Zero;
            c8a = Vector512<float>.Zero; c8b = Vector512<float>.Zero;
            c9a = Vector512<float>.Zero; c9b = Vector512<float>.Zero;
            c10a = Vector512<float>.Zero; c10b = Vector512<float>.Zero;
            c11a = Vector512<float>.Zero; c11b = Vector512<float>.Zero;
        }
        for (int p = 0; p < kc; p++)
        {
            var b0 = Vector512.Load(b); var b1 = Vector512.Load(b + 16);
            { var x = Vector512.Create(a[0]); c0a = Avx512F.FusedMultiplyAdd(x, b0, c0a); c0b = Avx512F.FusedMultiplyAdd(x, b1, c0b); }
            { var x = Vector512.Create(a[1]); c1a = Avx512F.FusedMultiplyAdd(x, b0, c1a); c1b = Avx512F.FusedMultiplyAdd(x, b1, c1b); }
            { var x = Vector512.Create(a[2]); c2a = Avx512F.FusedMultiplyAdd(x, b0, c2a); c2b = Avx512F.FusedMultiplyAdd(x, b1, c2b); }
            { var x = Vector512.Create(a[3]); c3a = Avx512F.FusedMultiplyAdd(x, b0, c3a); c3b = Avx512F.FusedMultiplyAdd(x, b1, c3b); }
            { var x = Vector512.Create(a[4]); c4a = Avx512F.FusedMultiplyAdd(x, b0, c4a); c4b = Avx512F.FusedMultiplyAdd(x, b1, c4b); }
            { var x = Vector512.Create(a[5]); c5a = Avx512F.FusedMultiplyAdd(x, b0, c5a); c5b = Avx512F.FusedMultiplyAdd(x, b1, c5b); }
            { var x = Vector512.Create(a[6]); c6a = Avx512F.FusedMultiplyAdd(x, b0, c6a); c6b = Avx512F.FusedMultiplyAdd(x, b1, c6b); }
            { var x = Vector512.Create(a[7]); c7a = Avx512F.FusedMultiplyAdd(x, b0, c7a); c7b = Avx512F.FusedMultiplyAdd(x, b1, c7b); }
            { var x = Vector512.Create(a[8]); c8a = Avx512F.FusedMultiplyAdd(x, b0, c8a); c8b = Avx512F.FusedMultiplyAdd(x, b1, c8b); }
            { var x = Vector512.Create(a[9]); c9a = Avx512F.FusedMultiplyAdd(x, b0, c9a); c9b = Avx512F.FusedMultiplyAdd(x, b1, c9b); }
            { var x = Vector512.Create(a[10]); c10a = Avx512F.FusedMultiplyAdd(x, b0, c10a); c10b = Avx512F.FusedMultiplyAdd(x, b1, c10b); }
            { var x = Vector512.Create(a[11]); c11a = Avx512F.FusedMultiplyAdd(x, b0, c11a); c11b = Avx512F.FusedMultiplyAdd(x, b1, c11b); }
            a += 12; b += 32;
        }
        Vector512.Store(c0a, c + 0 * ldc); Vector512.Store(c0b, c + 0 * ldc + 16);
        Vector512.Store(c1a, c + 1 * ldc); Vector512.Store(c1b, c + 1 * ldc + 16);
        Vector512.Store(c2a, c + 2 * ldc); Vector512.Store(c2b, c + 2 * ldc + 16);
        Vector512.Store(c3a, c + 3 * ldc); Vector512.Store(c3b, c + 3 * ldc + 16);
        Vector512.Store(c4a, c + 4 * ldc); Vector512.Store(c4b, c + 4 * ldc + 16);
        Vector512.Store(c5a, c + 5 * ldc); Vector512.Store(c5b, c + 5 * ldc + 16);
        Vector512.Store(c6a, c + 6 * ldc); Vector512.Store(c6b, c + 6 * ldc + 16);
        Vector512.Store(c7a, c + 7 * ldc); Vector512.Store(c7b, c + 7 * ldc + 16);
        Vector512.Store(c8a, c + 8 * ldc); Vector512.Store(c8b, c + 8 * ldc + 16);
        Vector512.Store(c9a, c + 9 * ldc); Vector512.Store(c9b, c + 9 * ldc + 16);
        Vector512.Store(c10a, c + 10 * ldc); Vector512.Store(c10b, c + 10 * ldc + 16);
        Vector512.Store(c11a, c + 11 * ldc); Vector512.Store(c11b, c + 11 * ldc + 16);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel256(float* a, float* b, int kc, float* c, long ldc, int mode, float* bias)
    {
        Vector256<float> c0a, c0b, c0c, c0d, c1a, c1b, c1c, c1d;
        if (mode >= InitLoad && mode != InitBias)
        {
            c0a = Vector256.Load(c + 0 * ldc + 0);
            c0b = Vector256.Load(c + 0 * ldc + 8);
            c0c = Vector256.Load(c + 0 * ldc + 16);
            c0d = Vector256.Load(c + 0 * ldc + 24);
            c1a = Vector256.Load(c + 1 * ldc + 0);
            c1b = Vector256.Load(c + 1 * ldc + 8);
            c1c = Vector256.Load(c + 1 * ldc + 16);
            c1d = Vector256.Load(c + 1 * ldc + 24);
            if (mode == InitLoadBias)
            {
                c0a += Vector256.Load(bias + 0);
                c0b += Vector256.Load(bias + 8);
                c0c += Vector256.Load(bias + 16);
                c0d += Vector256.Load(bias + 24);
                c1a += Vector256.Load(bias + 0);
                c1b += Vector256.Load(bias + 8);
                c1c += Vector256.Load(bias + 16);
                c1d += Vector256.Load(bias + 24);
            }
        }
        else if (mode == InitBias)
        {
            c0a = Vector256.Load(bias + 0);
            c0b = Vector256.Load(bias + 8);
            c0c = Vector256.Load(bias + 16);
            c0d = Vector256.Load(bias + 24);
            c1a = Vector256.Load(bias + 0);
            c1b = Vector256.Load(bias + 8);
            c1c = Vector256.Load(bias + 16);
            c1d = Vector256.Load(bias + 24);
        }
        else
        {
            c0a = Vector256<float>.Zero;
            c0b = Vector256<float>.Zero;
            c0c = Vector256<float>.Zero;
            c0d = Vector256<float>.Zero;
            c1a = Vector256<float>.Zero;
            c1b = Vector256<float>.Zero;
            c1c = Vector256<float>.Zero;
            c1d = Vector256<float>.Zero;
        }
        for (int p = 0; p < kc; p++)
        {
            var b0 = Vector256.Load(b); var b1 = Vector256.Load(b + 8); var b2 = Vector256.Load(b + 16); var b3 = Vector256.Load(b + 24);
            { var x = Vector256.Create(a[0]); c0a = Fma.MultiplyAdd(x, b0, c0a); c0b = Fma.MultiplyAdd(x, b1, c0b); c0c = Fma.MultiplyAdd(x, b2, c0c); c0d = Fma.MultiplyAdd(x, b3, c0d); }
            { var x = Vector256.Create(a[1]); c1a = Fma.MultiplyAdd(x, b0, c1a); c1b = Fma.MultiplyAdd(x, b1, c1b); c1c = Fma.MultiplyAdd(x, b2, c1c); c1d = Fma.MultiplyAdd(x, b3, c1d); }
            a += 2; b += 32;
        }
        Vector256.Store(c0a, c + 0 * ldc + 0);
        Vector256.Store(c0b, c + 0 * ldc + 8);
        Vector256.Store(c0c, c + 0 * ldc + 16);
        Vector256.Store(c0d, c + 0 * ldc + 24);
        Vector256.Store(c1a, c + 1 * ldc + 0);
        Vector256.Store(c1b, c + 1 * ldc + 8);
        Vector256.Store(c1c, c + 1 * ldc + 16);
        Vector256.Store(c1d, c + 1 * ldc + 24);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel512Bf16(float* a, ushort* b, int kc, float* c, long ldc, int mode, float* bias)
    {
        Vector512<float> c0a, c0b, c1a, c1b, c2a, c2b, c3a, c3b, c4a, c4b, c5a, c5b, c6a, c6b, c7a, c7b, c8a, c8b, c9a, c9b, c10a, c10b, c11a, c11b;
        if (mode == InitZero)
        {
            c0a = Vector512<float>.Zero; c0b = Vector512<float>.Zero;
            c1a = Vector512<float>.Zero; c1b = Vector512<float>.Zero;
            c2a = Vector512<float>.Zero; c2b = Vector512<float>.Zero;
            c3a = Vector512<float>.Zero; c3b = Vector512<float>.Zero;
            c4a = Vector512<float>.Zero; c4b = Vector512<float>.Zero;
            c5a = Vector512<float>.Zero; c5b = Vector512<float>.Zero;
            c6a = Vector512<float>.Zero; c6b = Vector512<float>.Zero;
            c7a = Vector512<float>.Zero; c7b = Vector512<float>.Zero;
            c8a = Vector512<float>.Zero; c8b = Vector512<float>.Zero;
            c9a = Vector512<float>.Zero; c9b = Vector512<float>.Zero;
            c10a = Vector512<float>.Zero; c10b = Vector512<float>.Zero;
            c11a = Vector512<float>.Zero; c11b = Vector512<float>.Zero;
        }
        else if (mode == InitBias)
        {
            var ba = Vector512.Load(bias); var bb = Vector512.Load(bias + 16);
            c0a = ba; c0b = bb;
            c1a = ba; c1b = bb;
            c2a = ba; c2b = bb;
            c3a = ba; c3b = bb;
            c4a = ba; c4b = bb;
            c5a = ba; c5b = bb;
            c6a = ba; c6b = bb;
            c7a = ba; c7b = bb;
            c8a = ba; c8b = bb;
            c9a = ba; c9b = bb;
            c10a = ba; c10b = bb;
            c11a = ba; c11b = bb;
        }
        else
        {
            c0a = Vector512.Load(c + 0 * ldc); c0b = Vector512.Load(c + 0 * ldc + 16);
            c1a = Vector512.Load(c + 1 * ldc); c1b = Vector512.Load(c + 1 * ldc + 16);
            c2a = Vector512.Load(c + 2 * ldc); c2b = Vector512.Load(c + 2 * ldc + 16);
            c3a = Vector512.Load(c + 3 * ldc); c3b = Vector512.Load(c + 3 * ldc + 16);
            c4a = Vector512.Load(c + 4 * ldc); c4b = Vector512.Load(c + 4 * ldc + 16);
            c5a = Vector512.Load(c + 5 * ldc); c5b = Vector512.Load(c + 5 * ldc + 16);
            c6a = Vector512.Load(c + 6 * ldc); c6b = Vector512.Load(c + 6 * ldc + 16);
            c7a = Vector512.Load(c + 7 * ldc); c7b = Vector512.Load(c + 7 * ldc + 16);
            c8a = Vector512.Load(c + 8 * ldc); c8b = Vector512.Load(c + 8 * ldc + 16);
            c9a = Vector512.Load(c + 9 * ldc); c9b = Vector512.Load(c + 9 * ldc + 16);
            c10a = Vector512.Load(c + 10 * ldc); c10b = Vector512.Load(c + 10 * ldc + 16);
            c11a = Vector512.Load(c + 11 * ldc); c11b = Vector512.Load(c + 11 * ldc + 16);
            if (mode == InitLoadBias)
            {
                var ba = Vector512.Load(bias); var bb = Vector512.Load(bias + 16);
                c0a += ba; c0b += bb;
                c1a += ba; c1b += bb;
                c2a += ba; c2b += bb;
                c3a += ba; c3b += bb;
                c4a += ba; c4b += bb;
                c5a += ba; c5b += bb;
                c6a += ba; c6b += bb;
                c7a += ba; c7b += bb;
                c8a += ba; c8b += bb;
                c9a += ba; c9b += bb;
                c10a += ba; c10b += bb;
                c11a += ba; c11b += bb;
            }
        }
        for (int p = 0; p < kc; p++)
        {
            var b0 = Bf16.Load512(b); var b1 = Bf16.Load512(b + 16);
            { var x = Vector512.Create(a[0]); c0a = Avx512F.FusedMultiplyAdd(x, b0, c0a); c0b = Avx512F.FusedMultiplyAdd(x, b1, c0b); }
            { var x = Vector512.Create(a[1]); c1a = Avx512F.FusedMultiplyAdd(x, b0, c1a); c1b = Avx512F.FusedMultiplyAdd(x, b1, c1b); }
            { var x = Vector512.Create(a[2]); c2a = Avx512F.FusedMultiplyAdd(x, b0, c2a); c2b = Avx512F.FusedMultiplyAdd(x, b1, c2b); }
            { var x = Vector512.Create(a[3]); c3a = Avx512F.FusedMultiplyAdd(x, b0, c3a); c3b = Avx512F.FusedMultiplyAdd(x, b1, c3b); }
            { var x = Vector512.Create(a[4]); c4a = Avx512F.FusedMultiplyAdd(x, b0, c4a); c4b = Avx512F.FusedMultiplyAdd(x, b1, c4b); }
            { var x = Vector512.Create(a[5]); c5a = Avx512F.FusedMultiplyAdd(x, b0, c5a); c5b = Avx512F.FusedMultiplyAdd(x, b1, c5b); }
            { var x = Vector512.Create(a[6]); c6a = Avx512F.FusedMultiplyAdd(x, b0, c6a); c6b = Avx512F.FusedMultiplyAdd(x, b1, c6b); }
            { var x = Vector512.Create(a[7]); c7a = Avx512F.FusedMultiplyAdd(x, b0, c7a); c7b = Avx512F.FusedMultiplyAdd(x, b1, c7b); }
            { var x = Vector512.Create(a[8]); c8a = Avx512F.FusedMultiplyAdd(x, b0, c8a); c8b = Avx512F.FusedMultiplyAdd(x, b1, c8b); }
            { var x = Vector512.Create(a[9]); c9a = Avx512F.FusedMultiplyAdd(x, b0, c9a); c9b = Avx512F.FusedMultiplyAdd(x, b1, c9b); }
            { var x = Vector512.Create(a[10]); c10a = Avx512F.FusedMultiplyAdd(x, b0, c10a); c10b = Avx512F.FusedMultiplyAdd(x, b1, c10b); }
            { var x = Vector512.Create(a[11]); c11a = Avx512F.FusedMultiplyAdd(x, b0, c11a); c11b = Avx512F.FusedMultiplyAdd(x, b1, c11b); }
            a += 12; b += 32;
        }
        Vector512.Store(c0a, c + 0 * ldc); Vector512.Store(c0b, c + 0 * ldc + 16);
        Vector512.Store(c1a, c + 1 * ldc); Vector512.Store(c1b, c + 1 * ldc + 16);
        Vector512.Store(c2a, c + 2 * ldc); Vector512.Store(c2b, c + 2 * ldc + 16);
        Vector512.Store(c3a, c + 3 * ldc); Vector512.Store(c3b, c + 3 * ldc + 16);
        Vector512.Store(c4a, c + 4 * ldc); Vector512.Store(c4b, c + 4 * ldc + 16);
        Vector512.Store(c5a, c + 5 * ldc); Vector512.Store(c5b, c + 5 * ldc + 16);
        Vector512.Store(c6a, c + 6 * ldc); Vector512.Store(c6b, c + 6 * ldc + 16);
        Vector512.Store(c7a, c + 7 * ldc); Vector512.Store(c7b, c + 7 * ldc + 16);
        Vector512.Store(c8a, c + 8 * ldc); Vector512.Store(c8b, c + 8 * ldc + 16);
        Vector512.Store(c9a, c + 9 * ldc); Vector512.Store(c9b, c + 9 * ldc + 16);
        Vector512.Store(c10a, c + 10 * ldc); Vector512.Store(c10b, c + 10 * ldc + 16);
        Vector512.Store(c11a, c + 11 * ldc); Vector512.Store(c11b, c + 11 * ldc + 16);
    }

    private static void KernelScalar(float* a, float* b, int kc, float* c, long ldc, int mode, float* bias)
    {
        // MR == 1 on this path.
        Span<float> acc = stackalloc float[NR];
        for (int j = 0; j < NR; j++) acc[j] = (mode == InitLoad || mode == InitLoadBias ? c[j] : 0f) + (mode == InitBias || mode == InitLoadBias ? bias[j] : 0f);
        for (int p = 0; p < kc; p++)
        {
            float x = a[p];
            float* bp = b + p * NR;
            for (int j = 0; j < NR; j++) acc[j] = MathF.FusedMultiplyAdd(x, bp[j], acc[j]);
        }
        for (int j = 0; j < NR; j++) c[j] = acc[j];
    }


    // ---------------------------------------------------------------------------------------------
    // Convolution GEMM for narrow outputs (N a multiple of 96, ≤ 288): a 4 × 96 register tile that reads
    // A straight from the padded channels-last input - output row t, tap τ, channel ci lives at
    // x[t·rowStride + τ·tapStride + ci] - so neither im2col nor A packing is needed; the B block
    // (64 × 96, 24 KB) stays in L1 across all row tiles of a chunk.
    // ---------------------------------------------------------------------------------------------

    private const int WideCols = 96, WideRows = 4, WideKc = 64, WideMc = 256;

    /// <summary>Whether <see cref="Conv"/> takes the narrow-output path for this weight.</summary>
    public static bool UseWideConv(PackedMatrix b) => UseAvx512 && !b.IsBf16 && !b.IsInt8 && b.N % WideCols == 0;

    /// <summary>
    /// <c>C[t, :] (+)= Σ_τ Σ_ci x[t·rowStride + τ·tapStride + ci] · B[τ·cin + ci, :] (+ bias)</c> for t &lt; m, where
    /// B has K = taps · cin. Falls back to the packed GEMM (with an im2col copy when taps overlap non-contiguously).
    /// </summary>
    public static void Conv(float* x, long rowStride, long tapStride, int taps, int cin, int m, PackedMatrix b, float* c, long ldc,
                            Parallelism par, float* bias = null, bool accumulate = false)
    {
        if (b.K != taps * cin) throw new ArgumentException("B depth must be taps × cin.");
        if (m <= 0) return;
        if (!UseWideConv(b))
        {
            if (tapStride == cin || taps == 1)
            {
                Multiply(x, rowStride, m, b, c, ldc, par, bias, accumulate);
                return;
            }
            int k = taps * cin;
            int tile = Math.Max(64, (1 << 18) / k);
            using var cols = new NativeBuffer<float>((long)tile * k, zero: false);
            for (int t0 = 0; t0 < m; t0 += tile)
            {
                int n = Math.Min(tile, m - t0);
                for (int r = 0; r < n; r++)
                    for (int tap = 0; tap < taps; tap++)
                        Buffer.MemoryCopy(x + (t0 + r) * rowStride + tap * tapStride, cols.Ptr + (long)r * k + tap * cin, cin * 4, cin * 4);
                Multiply(cols.Ptr, k, n, b, c + (long)t0 * ldc, ldc, par, bias, accumulate);
            }
            return;
        }
        int chunks = (m + WideMc - 1) / WideMc;
        par.For(chunks, ch =>
        {
            int m0 = ch * WideMc, m1 = Math.Min(m, m0 + WideMc);
            WideChunk(x, rowStride, tapStride, taps, cin, m0, m1, b, c, ldc, bias, accumulate);
        });
    }

    private static void WideChunk(float* x, long rowStride, long tapStride, int taps, int cin, int m0, int m1, PackedMatrix b,
                                  float* c, long ldc, float* bias, bool accumulate)
    {
        int K = b.K, colBlocks = b.N / WideCols;
        long panelStride = (long)K * NR;
        float* tailA = stackalloc float[WideRows * WideKc];
        float* tailC = stackalloc float[WideRows * WideCols];
        bool first = true;
        for (int tap = 0; tap < taps; tap++)
        {
            for (int ci0 = 0; ci0 < cin; ci0 += WideKc)
            {
                int kc = Math.Min(WideKc, cin - ci0);
                int kAbs = tap * cin + ci0;
                int mode = !first ? InitLoad : accumulate ? (bias != null ? InitLoadBias : InitLoad) : bias != null ? InitBias : InitZero;
                for (int cb = 0; cb < colBlocks; cb++)
                {
                    float* bp = b.F32 + ((long)(cb * 3) * K + kAbs) * NR;
                    float* biasp = bias == null ? null : bias + cb * WideCols;
                    int t = m0;
                    for (; t + WideRows <= m1; t += WideRows)
                        KernelWide(x + t * rowStride + tap * tapStride + ci0, rowStride, bp, panelStride, kc,
                                   c + t * ldc + cb * WideCols, ldc, mode, biasp);
                    if (t < m1)
                    {
                        int rows = m1 - t;
                        for (int r = 0; r < WideRows; r++)
                            for (int p = 0; p < kc; p++)
                                tailA[r * WideKc + p] = r < rows ? x[(t + r) * rowStride + tap * tapStride + ci0 + p] : 0f;
                        for (int r = 0; r < WideRows; r++)
                            for (int j = 0; j < WideCols; j++)
                                tailC[r * WideCols + j] = r < rows ? c[(t + r) * ldc + cb * WideCols + j] : 0f;
                        KernelWide(tailA, WideKc, bp, panelStride, kc, tailC, WideCols, mode, biasp);
                        for (int r = 0; r < rows; r++)
                            for (int j = 0; j < WideCols; j++) c[(t + r) * ldc + cb * WideCols + j] = tailC[r * WideCols + j];
                    }
                }
                first = false;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void KernelWide(float* a, long lda, float* b, long panelStride, int kc, float* c, long ldc, int mode, float* bias)
    {
        Vector512<float> c00, c01, c02, c03, c04, c05, c10, c11, c12, c13, c14, c15, c20, c21, c22, c23, c24, c25, c30, c31, c32, c33, c34, c35;
        if (mode == InitZero)
        {
            c00 = Vector512<float>.Zero; c01 = Vector512<float>.Zero; c02 = Vector512<float>.Zero; c03 = Vector512<float>.Zero; c04 = Vector512<float>.Zero; c05 = Vector512<float>.Zero;
            c10 = Vector512<float>.Zero; c11 = Vector512<float>.Zero; c12 = Vector512<float>.Zero; c13 = Vector512<float>.Zero; c14 = Vector512<float>.Zero; c15 = Vector512<float>.Zero;
            c20 = Vector512<float>.Zero; c21 = Vector512<float>.Zero; c22 = Vector512<float>.Zero; c23 = Vector512<float>.Zero; c24 = Vector512<float>.Zero; c25 = Vector512<float>.Zero;
            c30 = Vector512<float>.Zero; c31 = Vector512<float>.Zero; c32 = Vector512<float>.Zero; c33 = Vector512<float>.Zero; c34 = Vector512<float>.Zero; c35 = Vector512<float>.Zero;
        }
        else if (mode == InitBias)
        {
            c00 = Vector512.Load(bias + 0); c01 = Vector512.Load(bias + 16); c02 = Vector512.Load(bias + 32); c03 = Vector512.Load(bias + 48); c04 = Vector512.Load(bias + 64); c05 = Vector512.Load(bias + 80);
            c10 = Vector512.Load(bias + 0); c11 = Vector512.Load(bias + 16); c12 = Vector512.Load(bias + 32); c13 = Vector512.Load(bias + 48); c14 = Vector512.Load(bias + 64); c15 = Vector512.Load(bias + 80);
            c20 = Vector512.Load(bias + 0); c21 = Vector512.Load(bias + 16); c22 = Vector512.Load(bias + 32); c23 = Vector512.Load(bias + 48); c24 = Vector512.Load(bias + 64); c25 = Vector512.Load(bias + 80);
            c30 = Vector512.Load(bias + 0); c31 = Vector512.Load(bias + 16); c32 = Vector512.Load(bias + 32); c33 = Vector512.Load(bias + 48); c34 = Vector512.Load(bias + 64); c35 = Vector512.Load(bias + 80);
        }
        else
        {
            c00 = Vector512.Load(c + 0 * ldc + 0); c01 = Vector512.Load(c + 0 * ldc + 16); c02 = Vector512.Load(c + 0 * ldc + 32); c03 = Vector512.Load(c + 0 * ldc + 48); c04 = Vector512.Load(c + 0 * ldc + 64); c05 = Vector512.Load(c + 0 * ldc + 80);
            c10 = Vector512.Load(c + 1 * ldc + 0); c11 = Vector512.Load(c + 1 * ldc + 16); c12 = Vector512.Load(c + 1 * ldc + 32); c13 = Vector512.Load(c + 1 * ldc + 48); c14 = Vector512.Load(c + 1 * ldc + 64); c15 = Vector512.Load(c + 1 * ldc + 80);
            c20 = Vector512.Load(c + 2 * ldc + 0); c21 = Vector512.Load(c + 2 * ldc + 16); c22 = Vector512.Load(c + 2 * ldc + 32); c23 = Vector512.Load(c + 2 * ldc + 48); c24 = Vector512.Load(c + 2 * ldc + 64); c25 = Vector512.Load(c + 2 * ldc + 80);
            c30 = Vector512.Load(c + 3 * ldc + 0); c31 = Vector512.Load(c + 3 * ldc + 16); c32 = Vector512.Load(c + 3 * ldc + 32); c33 = Vector512.Load(c + 3 * ldc + 48); c34 = Vector512.Load(c + 3 * ldc + 64); c35 = Vector512.Load(c + 3 * ldc + 80);
            if (mode == InitLoadBias)
            {
                { var bv = Vector512.Load(bias + 0); c00 += bv; c10 += bv; c20 += bv; c30 += bv; }
                { var bv = Vector512.Load(bias + 16); c01 += bv; c11 += bv; c21 += bv; c31 += bv; }
                { var bv = Vector512.Load(bias + 32); c02 += bv; c12 += bv; c22 += bv; c32 += bv; }
                { var bv = Vector512.Load(bias + 48); c03 += bv; c13 += bv; c23 += bv; c33 += bv; }
                { var bv = Vector512.Load(bias + 64); c04 += bv; c14 += bv; c24 += bv; c34 += bv; }
                { var bv = Vector512.Load(bias + 80); c05 += bv; c15 += bv; c25 += bv; c35 += bv; }
            }
        }
        float* a1 = a + lda, a2 = a + 2 * lda, a3 = a + 3 * lda;
        float* b1 = b + panelStride, b2 = b + 2 * panelStride;
        for (int p = 0; p < kc; p++)
        {
            var x0 = Vector512.Create(a[p]); var x1 = Vector512.Create(a1[p]); var x2 = Vector512.Create(a2[p]); var x3 = Vector512.Create(a3[p]);
            { var w = Vector512.Load(b); c00 = Avx512F.FusedMultiplyAdd(x0, w, c00); c10 = Avx512F.FusedMultiplyAdd(x1, w, c10); c20 = Avx512F.FusedMultiplyAdd(x2, w, c20); c30 = Avx512F.FusedMultiplyAdd(x3, w, c30); }
            { var w = Vector512.Load(b + 16); c01 = Avx512F.FusedMultiplyAdd(x0, w, c01); c11 = Avx512F.FusedMultiplyAdd(x1, w, c11); c21 = Avx512F.FusedMultiplyAdd(x2, w, c21); c31 = Avx512F.FusedMultiplyAdd(x3, w, c31); }
            { var w = Vector512.Load(b1); c02 = Avx512F.FusedMultiplyAdd(x0, w, c02); c12 = Avx512F.FusedMultiplyAdd(x1, w, c12); c22 = Avx512F.FusedMultiplyAdd(x2, w, c22); c32 = Avx512F.FusedMultiplyAdd(x3, w, c32); }
            { var w = Vector512.Load(b1 + 16); c03 = Avx512F.FusedMultiplyAdd(x0, w, c03); c13 = Avx512F.FusedMultiplyAdd(x1, w, c13); c23 = Avx512F.FusedMultiplyAdd(x2, w, c23); c33 = Avx512F.FusedMultiplyAdd(x3, w, c33); }
            { var w = Vector512.Load(b2); c04 = Avx512F.FusedMultiplyAdd(x0, w, c04); c14 = Avx512F.FusedMultiplyAdd(x1, w, c14); c24 = Avx512F.FusedMultiplyAdd(x2, w, c24); c34 = Avx512F.FusedMultiplyAdd(x3, w, c34); }
            { var w = Vector512.Load(b2 + 16); c05 = Avx512F.FusedMultiplyAdd(x0, w, c05); c15 = Avx512F.FusedMultiplyAdd(x1, w, c15); c25 = Avx512F.FusedMultiplyAdd(x2, w, c25); c35 = Avx512F.FusedMultiplyAdd(x3, w, c35); }
            b += 32; b1 += 32; b2 += 32;
        }
        Vector512.Store(c00, c + 0 * ldc + 0); Vector512.Store(c01, c + 0 * ldc + 16); Vector512.Store(c02, c + 0 * ldc + 32); Vector512.Store(c03, c + 0 * ldc + 48); Vector512.Store(c04, c + 0 * ldc + 64); Vector512.Store(c05, c + 0 * ldc + 80);
        Vector512.Store(c10, c + 1 * ldc + 0); Vector512.Store(c11, c + 1 * ldc + 16); Vector512.Store(c12, c + 1 * ldc + 32); Vector512.Store(c13, c + 1 * ldc + 48); Vector512.Store(c14, c + 1 * ldc + 64); Vector512.Store(c15, c + 1 * ldc + 80);
        Vector512.Store(c20, c + 2 * ldc + 0); Vector512.Store(c21, c + 2 * ldc + 16); Vector512.Store(c22, c + 2 * ldc + 32); Vector512.Store(c23, c + 2 * ldc + 48); Vector512.Store(c24, c + 2 * ldc + 64); Vector512.Store(c25, c + 2 * ldc + 80);
        Vector512.Store(c30, c + 3 * ldc + 0); Vector512.Store(c31, c + 3 * ldc + 16); Vector512.Store(c32, c + 3 * ldc + 32); Vector512.Store(c33, c + 3 * ldc + 48); Vector512.Store(c34, c + 3 * ldc + 64); Vector512.Store(c35, c + 3 * ldc + 80);
    }

    // ---------------------------------------------------------------------------------------------
    // Small-m path (decode): every weight panel is streamed once and dotted against ≤ 4 rows.
    // ---------------------------------------------------------------------------------------------

    private static void SmallM(float* a, long lda, int m, PackedMatrix b, float* c, long ldc, Parallelism par, float* bias, bool accumulate)
    {
        int blocks = b.Blocks;
        par.ForRange(blocks, 8, (nb0, nb1) =>
        {
            float* tile = stackalloc float[4 * NR];
            for (int nb = nb0; nb < nb1; nb++)
            {
                if (m == 1 && nb + 1 < nb1 && UseAvx512)
                {
                    // Two weight panels streamed together: more misses in flight for the bandwidth-bound decode.
                    if (b.IsInt8) PanelPairDotInt8(a, b, nb, tile);
                    else if (b.IsBf16) PanelPairDotBf16(a, b, nb, tile);
                    else goto single;
                    for (int p2 = 0; p2 < 2; p2++)
                    {
                        int nbi = nb + p2;
                        int cols2 = Math.Min(NR, b.N - nbi * NR);
                        float* cp = c + nbi * NR;
                        float* tp = tile + p2 * NR;
                        if (accumulate && bias != null) for (int j = 0; j < cols2; j++) cp[j] += tp[j] + bias[nbi * NR + j];
                        else if (accumulate) for (int j = 0; j < cols2; j++) cp[j] += tp[j];
                        else if (bias != null) for (int j = 0; j < cols2; j++) cp[j] = tp[j] + bias[nbi * NR + j];
                        else for (int j = 0; j < cols2; j++) cp[j] = tp[j];
                    }
                    nb++;
                    continue;
                }
                single:
                for (int r = 0; r < m; r += 2)
                {
                    int mm = Math.Min(2, m - r);
                    PanelDot(a + r * lda, lda, mm, b, nb, tile);
                    int cols = Math.Min(NR, b.N - nb * NR);
                    for (int i = 0; i < mm; i++)
                    {
                        float* cp = c + (r + i) * ldc + nb * NR;
                        float* tp = tile + i * NR;
                        if (accumulate && bias != null) for (int j = 0; j < cols; j++) cp[j] += tp[j] + bias[nb * NR + j];
                        else if (accumulate) for (int j = 0; j < cols; j++) cp[j] += tp[j];
                        else if (bias != null) for (int j = 0; j < cols; j++) cp[j] = tp[j] + bias[nb * NR + j];
                        else for (int j = 0; j < cols; j++) cp[j] = tp[j];
                    }
                }
            }
        });
    }

    private static void DequantizeInt8(sbyte* q, float* scale, float* dst, int rows)
    {
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < NR; j++) dst[r * NR + j] = q[r * NR + j] * scale[j];
    }

    private static Vector512<float> I8(sbyte* p) => Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(p)));

    /// <summary>m = 1 dot of x against panels nb and nb + 1 (int8), streamed concurrently.</summary>
    private static void PanelPairDotInt8(float* x, PackedMatrix b, int nb, float* outTile)
    {
        int K = b.K;
        sbyte* w = b.I8 + (long)nb * K * NR;
        sbyte* v = w + (long)K * NR;
        Vector512<float> a0 = default, a1 = default, c0 = default, c1 = default;
        for (int k = 0; k < K; k++)
        {
            var xk = Vector512.Create(x[k]);
            a0 = Avx512F.FusedMultiplyAdd(xk, I8(w), a0);
            a1 = Avx512F.FusedMultiplyAdd(xk, I8(w + 16), a1);
            c0 = Avx512F.FusedMultiplyAdd(xk, I8(v), c0);
            c1 = Avx512F.FusedMultiplyAdd(xk, I8(v + 16), c1);
            w += 32;
            v += 32;
        }
        float* sc = b.Scale + nb * NR;
        Vector512.Store(a0 * Vector512.Load(sc), outTile);
        Vector512.Store(a1 * Vector512.Load(sc + 16), outTile + 16);
        Vector512.Store(c0 * Vector512.Load(sc + 32), outTile + 32);
        Vector512.Store(c1 * Vector512.Load(sc + 48), outTile + 48);
    }

    /// <summary>m = 1 dot of x against panels nb and nb + 1 (bf16), streamed concurrently.</summary>
    private static void PanelPairDotBf16(float* x, PackedMatrix b, int nb, float* outTile)
    {
        int K = b.K;
        ushort* w = b.B16 + (long)nb * K * NR;
        ushort* v = w + (long)K * NR;
        Vector512<float> a0 = default, a1 = default, c0 = default, c1 = default;
        for (int k = 0; k < K; k++)
        {
            var xk = Vector512.Create(x[k]);
            a0 = Avx512F.FusedMultiplyAdd(xk, Bf16.Load512(w), a0);
            a1 = Avx512F.FusedMultiplyAdd(xk, Bf16.Load512(w + 16), a1);
            c0 = Avx512F.FusedMultiplyAdd(xk, Bf16.Load512(v), c0);
            c1 = Avx512F.FusedMultiplyAdd(xk, Bf16.Load512(v + 16), c1);
            w += 32;
            v += 32;
        }
        Vector512.Store(a0, outTile);
        Vector512.Store(a1, outTile + 16);
        Vector512.Store(c0, outTile + 32);
        Vector512.Store(c1, outTile + 48);
    }

    /// <summary>out[i, 0..32) = scale · Σ_k x_i[k] · q[k, 0..32) for an int8 panel, i &lt; m ≤ 2.</summary>
    private static void PanelDotInt8(float* x, long ldx, int m, PackedMatrix b, int nb, float* outTile)
    {
        int K = b.K;
        sbyte* w = b.I8 + (long)nb * K * NR;
        float* sc = b.Scale + nb * NR;
        if (UseAvx512)
        {
            Vector512<float> a0 = default, a1 = default, b0 = default, b1 = default, a2 = default, a3 = default;
            float* y = m > 1 ? x + ldx : x;
            int k = 0;
            if (m == 1)
            {
                for (; k <= K - 2; k += 2)
                {
                    var x0 = Vector512.Create(x[k]); var x1 = Vector512.Create(x[k + 1]);
                    a0 = Avx512F.FusedMultiplyAdd(x0, Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(w))), a0);
                    a1 = Avx512F.FusedMultiplyAdd(x0, Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(w + 16))), a1);
                    a2 = Avx512F.FusedMultiplyAdd(x1, Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(w + 32))), a2);
                    a3 = Avx512F.FusedMultiplyAdd(x1, Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(w + 48))), a3);
                    w += 64;
                }
                a0 += a2;
                a1 += a3;
            }
            for (; k < K; k++)
            {
                var w0 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(w)));
                var w1 = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(w + 16)));
                var x0 = Vector512.Create(x[k]); var y0 = Vector512.Create(y[k]);
                a0 = Avx512F.FusedMultiplyAdd(x0, w0, a0);
                a1 = Avx512F.FusedMultiplyAdd(x0, w1, a1);
                b0 = Avx512F.FusedMultiplyAdd(y0, w0, b0);
                b1 = Avx512F.FusedMultiplyAdd(y0, w1, b1);
                w += 32;
            }
            var s0 = Vector512.Load(sc); var s1 = Vector512.Load(sc + 16);
            Vector512.Store(a0 * s0, outTile); Vector512.Store(a1 * s1, outTile + 16);
            Vector512.Store(b0 * s0, outTile + NR); Vector512.Store(b1 * s1, outTile + NR + 16);
            return;
        }
        for (int i = 0; i < m; i++)
        {
            float* xi = x + i * ldx;
            float* ot = outTile + i * NR;
            for (int j = 0; j < NR; j++) ot[j] = 0f;
            for (int kk = 0; kk < K; kk++)
                for (int j = 0; j < NR; j++) ot[j] = MathF.FusedMultiplyAdd(xi[kk], w[(long)kk * NR + j], ot[j]);
            for (int j = 0; j < NR; j++) ot[j] *= sc[j];
        }
    }

    /// <summary>out[i, 0..32) = Σ_k x_i[k] · panel[k, 0..32) for i &lt; m ≤ 2.</summary>
    private static void PanelDot(float* x, long ldx, int m, PackedMatrix b, int nb, float* outTile)
    {
        if (b.IsInt8)
        {
            PanelDotInt8(x, ldx, m, b, nb, outTile);
            return;
        }
        int K = b.K;
        long off = (long)nb * K * NR;
        if (UseAvx512)
        {
            if (b.IsBf16)
            {
                ushort* w = b.B16 + off;
                if (m == 1)
                {
                    Vector512<float> a0 = default, a1 = default, a2 = default, a3 = default;
                    int k = 0;
                    for (; k <= K - 2; k += 2)
                    {
                        var x0 = Vector512.Create(x[k]); var x1 = Vector512.Create(x[k + 1]);
                        a0 = Avx512F.FusedMultiplyAdd(x0, Bf16.Load512(w), a0);
                        a1 = Avx512F.FusedMultiplyAdd(x0, Bf16.Load512(w + 16), a1);
                        a2 = Avx512F.FusedMultiplyAdd(x1, Bf16.Load512(w + 32), a2);
                        a3 = Avx512F.FusedMultiplyAdd(x1, Bf16.Load512(w + 48), a3);
                        w += 64;
                    }
                    for (; k < K; k++)
                    {
                        var x0 = Vector512.Create(x[k]);
                        a0 = Avx512F.FusedMultiplyAdd(x0, Bf16.Load512(w), a0);
                        a1 = Avx512F.FusedMultiplyAdd(x0, Bf16.Load512(w + 16), a1);
                        w += 32;
                    }
                    Vector512.Store(a0 + a2, outTile);
                    Vector512.Store(a1 + a3, outTile + 16);
                }
                else
                {
                    Vector512<float> a0 = default, a1 = default, b0 = default, b1 = default;
                    float* y = x + ldx;
                    for (int k = 0; k < K; k++)
                    {
                        var w0 = Bf16.Load512(w); var w1 = Bf16.Load512(w + 16);
                        var x0 = Vector512.Create(x[k]); var y0 = Vector512.Create(y[k]);
                        a0 = Avx512F.FusedMultiplyAdd(x0, w0, a0);
                        a1 = Avx512F.FusedMultiplyAdd(x0, w1, a1);
                        b0 = Avx512F.FusedMultiplyAdd(y0, w0, b0);
                        b1 = Avx512F.FusedMultiplyAdd(y0, w1, b1);
                        w += 32;
                    }
                    Vector512.Store(a0, outTile); Vector512.Store(a1, outTile + 16);
                    Vector512.Store(b0, outTile + NR); Vector512.Store(b1, outTile + NR + 16);
                }
            }
            else
            {
                float* w = b.F32 + off;
                Vector512<float> a0 = default, a1 = default, b0 = default, b1 = default;
                float* y = m > 1 ? x + ldx : x;
                for (int k = 0; k < K; k++)
                {
                    var w0 = Vector512.Load(w); var w1 = Vector512.Load(w + 16);
                    var x0 = Vector512.Create(x[k]); var y0 = Vector512.Create(y[k]);
                    a0 = Avx512F.FusedMultiplyAdd(x0, w0, a0);
                    a1 = Avx512F.FusedMultiplyAdd(x0, w1, a1);
                    b0 = Avx512F.FusedMultiplyAdd(y0, w0, b0);
                    b1 = Avx512F.FusedMultiplyAdd(y0, w1, b1);
                    w += 32;
                }
                Vector512.Store(a0, outTile); Vector512.Store(a1, outTile + 16);
                Vector512.Store(b0, outTile + NR); Vector512.Store(b1, outTile + NR + 16);
            }
            return;
        }
        if (UseAvx2)
        {
            for (int i = 0; i < m; i++)
            {
                float* xi = x + i * ldx;
                Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default;
                for (int k = 0; k < K; k++)
                {
                    var xv = Vector256.Create(xi[k]);
                    long o = off + (long)k * NR;
                    if (b.IsBf16)
                    {
                        ushort* w = b.B16 + o;
                        a0 = Fma.MultiplyAdd(xv, Bf16.Load256(w), a0);
                        a1 = Fma.MultiplyAdd(xv, Bf16.Load256(w + 8), a1);
                        a2 = Fma.MultiplyAdd(xv, Bf16.Load256(w + 16), a2);
                        a3 = Fma.MultiplyAdd(xv, Bf16.Load256(w + 24), a3);
                    }
                    else
                    {
                        float* w = b.F32 + o;
                        a0 = Fma.MultiplyAdd(xv, Vector256.Load(w), a0);
                        a1 = Fma.MultiplyAdd(xv, Vector256.Load(w + 8), a1);
                        a2 = Fma.MultiplyAdd(xv, Vector256.Load(w + 16), a2);
                        a3 = Fma.MultiplyAdd(xv, Vector256.Load(w + 24), a3);
                    }
                }
                float* ot = outTile + i * NR;
                Vector256.Store(a0, ot); Vector256.Store(a1, ot + 8); Vector256.Store(a2, ot + 16); Vector256.Store(a3, ot + 24);
            }
            return;
        }
        for (int i = 0; i < m; i++)
        {
            float* xi = x + i * ldx;
            float* ot = outTile + i * NR;
            for (int j = 0; j < NR; j++) ot[j] = 0f;
            for (int k = 0; k < K; k++)
            {
                long o = off + (long)k * NR;
                for (int j = 0; j < NR; j++)
                {
                    float wv = b.IsBf16 ? Bf16.ToFloat(b.B16[o + j]) : b.F32[o + j];
                    ot[j] = MathF.FusedMultiplyAdd(xi[k], wv, ot[j]);
                }
            }
        }
    }
}
