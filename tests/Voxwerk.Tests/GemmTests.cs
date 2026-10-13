using System.Diagnostics;
using Voxwerk.Numerics;
using Xunit.Abstractions;

namespace Voxwerk.Tests;

public unsafe class GemmTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(1, 70, 33, false)]
    [InlineData(2, 64, 300, true)]
    [InlineData(3, 100, 17, false)]
    [InlineData(13, 65, 257, false)]
    [InlineData(250, 96, 515, true)]
    [InlineData(37, 31, 1, false)]
    public void MatchesNaive(int m, int n, int k, bool bf16)
    {
        var rng = new Random(42 + m + n + k);
        var a = new float[m * k];
        var w = new float[n * k];
        var bias = new float[n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < n; i++) bias[i] = (float)rng.NextDouble();
        PackedMatrix pm;
        if (bf16)
        {
            using var wb = new NativeBuffer<ushort>(w.Length);
            for (int i = 0; i < w.Length; i++) { wb.Ptr[i] = Bf16.FromFloat(w[i]); w[i] = Bf16.ToFloat(wb.Ptr[i]); }
            pm = PackedMatrix.FromBf16Linear(k, wb);
        }
        else
        {
            fixed (float* wp = w) pm = PackedMatrix.FromLinear(wp, n, k);
        }
        using (pm)
        {
            foreach (var threads in new[] { 1, 3 })
            {
                var c = new float[m * n];
                var c2 = new float[m * n];
                fixed (float* ap = a, cp = c, c2p = c2, bp = bias)
                {
                    Gemm.Multiply(ap, k, m, pm, cp, n, new Parallelism(threads), bp);
                    for (int i = 0; i < c2.Length; i++) c2[i] = 1f;
                    Gemm.Multiply(ap, k, m, pm, c2p, n, new Parallelism(threads), bp, accumulate: true);
                }
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        double s = 0;
                        for (int p = 0; p < k; p++) s += (double)a[i * k + p] * w[j * k + p];
                        Assert.True(Math.Abs(s + bias[j] - c[i * n + j]) < 1e-3, $"bias {i},{j}: {s + bias[j]} vs {c[i * n + j]}");
                        Assert.True(Math.Abs(s + 1 + bias[j] - c2[i * n + j]) < 1e-3, $"acc {i},{j}");
                    }
            }
        }
    }

    [Fact]
    public void Throughput()
    {
        int m = 512, n = 1024, k = 1024;
        var a = new float[m * k];
        var w = new float[n * k];
        new Random(1).NextBytes(new byte[1]);
        for (int i = 0; i < a.Length; i++) a[i] = i % 7 * 0.1f;
        for (int i = 0; i < w.Length; i++) w[i] = i % 5 * 0.1f;
        fixed (float* wp = w, ap = a)
        {
            using var pm = PackedMatrix.FromLinear(wp, n, k);
            var c = new float[m * n];
            fixed (float* cp = c)
            {
                Gemm.Multiply(ap, k, m, pm, cp, n, Parallelism.Single);
                var sw = Stopwatch.StartNew();
                int reps = 10;
                for (int r = 0; r < reps; r++) Gemm.Multiply(ap, k, m, pm, cp, n, Parallelism.Single);
                double gflops = 2.0 * m * n * k * reps / sw.Elapsed.TotalSeconds / 1e9;
                output.WriteLine($"single-thread sgemm {m}x{n}x{k}: {gflops:F1} GFLOP/s (MR={Gemm.MR})");
            }
        }
    }
}

public unsafe class ConvTests
{
    [Theory]
    [InlineData(103, 96, 96, 7, 1, 1)]
    [InlineData(57, 96, 96, 7, 9, 1)]
    [InlineData(64, 192, 192, 3, 3, 1)]
    [InlineData(41, 64, 288, 2, 1, 1)]
    [InlineData(50, 32, 64, 4, 1, 2)]
    [InlineData(33, 40, 96, 5, 2, 3)]
    public void ConvMatchesNaive(int tout, int cin, int cout, int taps, int dil, int stride)
    {
        var rng = new Random(tout + cin + cout);
        int tin = (tout - 1) * stride + (taps - 1) * dil + 1;
        var x = new float[tin * cin];
        var w = new float[cout * cin * taps];
        var bias = new float[cout];
        for (int i = 0; i < x.Length; i++) x[i] = (float)rng.NextDouble() - 0.5f;
        for (int i = 0; i < w.Length; i++) w[i] = (float)rng.NextDouble() - 0.5f;
        for (int i = 0; i < cout; i++) bias[i] = (float)rng.NextDouble();
        using var pm = fixedPack(w, cout, cin, taps);
        var y = new float[tout * cout];
        for (int i = 0; i < y.Length; i++) y[i] = 0.25f;
        fixed (float* xp = x, yp = y, bp = bias)
            Gemm.Conv(xp, (long)stride * cin, (long)dil * cin, taps, cin, tout, pm, yp, cout, new Parallelism(2), bp, accumulate: true);
        for (int t = 0; t < tout; t++)
            for (int co = 0; co < cout; co++)
            {
                double s = 0.25 + bias[co];
                for (int tap = 0; tap < taps; tap++)
                    for (int ci = 0; ci < cin; ci++)
                        s += (double)x[(t * stride + tap * dil) * cin + ci] * w[(co * cin + ci) * taps + tap];
                Assert.True(Math.Abs(s - y[t * cout + co]) < 1e-3, $"t={t} co={co}: {s} vs {y[t * cout + co]}");
            }
    }

    private static PackedMatrix fixedPack(float[] w, int cout, int cin, int taps)
    {
        fixed (float* wp = w) return PackedMatrix.FromConv(wp, cout, cin, taps);
    }
}
