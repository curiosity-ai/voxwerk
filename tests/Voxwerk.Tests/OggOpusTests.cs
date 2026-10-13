using Voxwerk.Audio;

namespace Voxwerk.Tests;

public class OggOpusTests
{
    [Fact]
    public void RoundTripPreservesLengthAndSignal()
    {
        int n = 24000 * 2 + 123;
        var pcm = new float[n];
        for (int i = 0; i < n; i++) pcm[i] = 0.3f * MathF.Sin(2 * MathF.PI * 220 * i / 24000f) + 0.1f * MathF.Sin(2 * MathF.PI * 1330 * i / 24000f);
        var bytes = OggOpus.Encode(pcm, tags: new Dictionary<string, string> { ["title"] = "test" });
        Assert.True(bytes.Length < n * 2 / 8, $"{bytes.Length} bytes is not compressed");
        Assert.Equal("OggS", System.Text.Encoding.ASCII.GetString(bytes, 0, 4));
        var back = OggOpus.Decode(bytes);
        Assert.Equal(n, back.Length);
        double num = 0, den = 0;
        for (int i = 2400; i < n - 2400; i++)
        {
            num += (pcm[i] - back[i]) * (pcm[i] - back[i]);
            den += pcm[i] * pcm[i];
        }
        double snr = 10 * Math.Log10(den / num);
        Assert.True(snr > 10, $"SNR {snr:F1} dB");
    }
}
