using Xunit.Abstractions;

namespace Voxwerk.Tests;

/// <summary>
/// Locates the model directory (<c>VOXWERK_MODEL_DIR</c>) and the reference tensors dumped by
/// <c>scripts/dump_reference.py</c> (<c>VOXWERK_REFERENCE_DIR</c>). Parity tests return early (and say so)
/// when either is missing, so the unit tests still run on a machine without the 2.5 GB checkpoint.
/// </summary>
public static class Fixtures
{
    public static string? ModelDir => Existing(Environment.GetEnvironmentVariable("VOXWERK_MODEL_DIR"));
    public static string? ReferenceDir => Existing(Environment.GetEnvironmentVariable("VOXWERK_REFERENCE_DIR"));

    private static string? Existing(string? p) => !string.IsNullOrEmpty(p) && Directory.Exists(p) ? p : null;

    public static bool Available(ITestOutputHelper output, bool needModel = true)
    {
        if (ReferenceDir == null || (needModel && ModelDir == null))
        {
            output.WriteLine("SKIPPED: set VOXWERK_MODEL_DIR and VOXWERK_REFERENCE_DIR (see scripts/dump_reference.py).");
            return false;
        }
        return true;
    }

    public static Npy Ref(string name) => Npy.Load(Path.Combine(ReferenceDir!, name + ".npy"));

    /// <summary>Asserts closeness and reports max-abs / relative-L2 error.</summary>
    public static void AssertClose(ITestOutputHelper output, string label, ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, double maxRelL2, double maxAbs = double.PositiveInfinity)
    {
        Assert.Equal(expected.Length, actual.Length);
        double num = 0, den = 0, mx = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = (double)expected[i] - actual[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
            mx = Math.Max(mx, Math.Abs(d));
        }
        double rel = Math.Sqrt(num / Math.Max(den, 1e-30));
        output.WriteLine($"{label,-32} n={expected.Length,9} maxAbs={mx:E2} relL2={rel:E2}");
        Assert.True(rel <= maxRelL2, $"{label}: relative L2 error {rel:E3} > {maxRelL2:E1}");
        Assert.True(mx <= maxAbs, $"{label}: max abs error {mx:E3} > {maxAbs:E1}");
    }
}
