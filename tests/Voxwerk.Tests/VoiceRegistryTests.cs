using Voxwerk.Voices;
using Xunit.Abstractions;

namespace Voxwerk.Tests;

public class VoiceRegistryTests(ITestOutputHelper output)
{
    private static readonly IVoicePack Pack = new EmbeddedVoicePack(typeof(VoiceRegistryTests).Assembly, "Voxwerk.Tests.TestVoices");

    [Fact]
    public void EmbeddedPackExposesMetadataAndBothSamples()
    {
        var v = Assert.Single(Pack.Voices);
        Assert.Equal("en-us-amber-hollis", v.Id);
        Assert.Equal("Voxwerk.Tests.TestVoices", v.Package);
        Assert.StartsWith("Hi, I'm Amber.", v.SampleText);
        Assert.False(string.IsNullOrEmpty(v.CloneText));

        var design = Pack.GetVoiceSample(v.Id)!;
        Assert.Equal(VoiceSampleKind.Design, design.Kind);
        Assert.Equal(24000, design.SampleRate);
        Assert.InRange(design.Audio.Length / 24000.0, v.SampleSeconds - 0.05, v.SampleSeconds + 0.05);
        var clone = Pack.GetVoiceSample(v.Id, VoiceSampleKind.Clone)!;
        Assert.Equal(v.CloneText, clone.Text);
        Assert.InRange(clone.Audio.Length / 24000.0, v.CloneSeconds - 0.05, v.CloneSeconds + 0.05);
        double rms = Math.Sqrt(design.Audio.Average(x => (double)x * x));
        output.WriteLine($"design {design.Audio.Length / 24000.0:F2}s rms {rms:F3}, clone {clone.Audio.Length / 24000.0:F2}s");
        Assert.True(rms > 0.01, "decoded sample is silent");

        using var s = Pack.OpenSampleStream(v.Id)!;
        var magic = new byte[4];
        s.ReadExactly(magic);
        Assert.Equal("OggS", System.Text.Encoding.ASCII.GetString(magic));
        Assert.Null(Pack.GetVoiceInfo("en-us-nobody"));
    }

    [Fact]
    public void RegistrySearchesAndFilters()
    {
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
        Assert.NotNull(VoiceRegistry.GetVoiceSample("en-us-amber-hollis", VoiceSampleKind.Clone));
    }
}
