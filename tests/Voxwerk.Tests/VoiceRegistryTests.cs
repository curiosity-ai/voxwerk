using Voxwerk.Voices;
using Xunit.Abstractions;

namespace Voxwerk.Tests;

public class VoiceRegistryTests(ITestOutputHelper output)
{
    private static readonly IVoicePack Pack = new EmbeddedVoicePack(typeof(VoiceRegistryTests).Assembly, "Voxwerk.Tests.TestVoices");
    private static readonly IVoiceSamplePack Samples = new EmbeddedVoiceSamplePack(typeof(VoiceRegistryTests).Assembly, "Voxwerk.Tests.TestVoices");

    [Fact]
    public void EmbeddedPacksExposeMetadataAndSamples()
    {
        var v = Assert.Single(Pack.Voices);
        Assert.Equal("en-us-amber-hollis", v.Id);
        Assert.Equal("Voxwerk.Tests.TestVoices", v.Package);
        Assert.StartsWith("Hi, I'm Amber.", v.SampleText);
        Assert.False(string.IsNullOrEmpty(v.DesignPrompt));
        Assert.Null(Pack.GetVoiceInfo("en-us-nobody"));

        Assert.Equal(["en-us-amber-hollis"], Samples.VoiceIds);
        using var s = Samples.OpenSampleStream(v.Id)!;
        var magic = new byte[4];
        s.ReadExactly(magic);
        Assert.Equal("OggS", System.Text.Encoding.ASCII.GetString(magic));
        Assert.Null(Samples.OpenSampleStream("en-us-nobody"));
    }

    [Fact]
    public void RegistrySearchesFiltersAndDecodesSamples()
    {
        VoiceRegistry.Clear();
        VoiceRegistry.Register(Pack);
        VoiceRegistry.Register(Pack);   // idempotent
        Assert.True(VoiceRegistry.Contains("en-us-amber-hollis"));
        Assert.NotNull(VoiceRegistry.GetVoiceInfo("EN-US-AMBER-HOLLIS"));
        Assert.Contains(VoiceRegistry.Search("warm southern"), v => v.Id == "en-us-amber-hollis");
        Assert.Contains(VoiceRegistry.Search(country: "us", gender: "female", category: "southern"), v => v.Id == "en-us-amber-hollis");
        Assert.DoesNotContain(VoiceRegistry.Search(country: "gb"), v => v.Id == "en-us-amber-hollis");
        Assert.Contains(VoiceRegistry.Search(style: "conversational"), v => v.Id == "en-us-amber-hollis");
        Assert.DoesNotContain(VoiceRegistry.Search(style: "narration"), v => v.Id == "en-us-amber-hollis");
        Assert.DoesNotContain(VoiceRegistry.Search("baritone"), v => v.Id == "en-us-amber-hollis");

        // Metadata alone: the voice is known but has no audio until its sample pack is registered.
        Assert.False(VoiceRegistry.HasSample("en-us-amber-hollis"));
        Assert.Null(VoiceRegistry.GetVoiceSample("en-us-amber-hollis"));
        VoiceRegistry.Register(Samples);
        Assert.True(VoiceRegistry.HasSample("en-us-amber-hollis"));
        var v = VoiceRegistry.GetVoiceInfo("en-us-amber-hollis")!;
        var sample = VoiceRegistry.GetVoiceSample(v.Id)!;
        Assert.Equal(24000, sample.SampleRate);
        Assert.Equal(v.SampleText, sample.Text);
        Assert.InRange(sample.Audio.Length / 24000.0, v.SampleSeconds - 0.05, v.SampleSeconds + 0.05);
        double rms = Math.Sqrt(sample.Audio.Average(x => (double)x * x));
        output.WriteLine($"sample {sample.Audio.Length / 24000.0:F2}s rms {rms:F3}");
        Assert.True(rms > 0.01, "decoded sample is silent");
    }
}
