using NAudio.Wave;
using Voxwerk.Audio;

namespace Voxwerk.Tests;

public class AudioFileTests
{
    [Fact]
    public void ReadsStereo44kAsMono24k()
    {
        string path = Path.Combine(Path.GetTempPath(), $"voxwerk-test-{Guid.NewGuid():N}.wav");
        try
        {
            const int rate = 44100;
            var format = new WaveFormat(rate, 16, 2);
            using (var w = new WaveFileWriter(path, format))
            {
                var pcm = new short[rate * 2];   // 1 s stereo, 440 Hz tone on the left channel only
                for (int i = 0; i < rate; i++) pcm[2 * i] = (short)(16000 * Math.Sin(2 * Math.PI * 440 * i / rate));
                w.WriteSamples(pcm, 0, pcm.Length);
            }
            var mono = AudioFile.ReadMono(path, 24000);
            Assert.InRange(mono.Length, 23900, 24100);
            double rms = Math.Sqrt(mono.Average(v => (double)v * v));
            // Mixdown halves the amplitude: 16000/32768 * 0.5 / sqrt(2) ≈ 0.173
            Assert.InRange(rms, 0.16, 0.19);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void WritesPcm16AndFloat()
    {
        string path = Path.Combine(Path.GetTempPath(), $"voxwerk-test-{Guid.NewGuid():N}.wav");
        try
        {
            var samples = Enumerable.Range(0, 2400).Select(i => 0.5f * MathF.Sin(i * 0.1f)).ToArray();
            foreach (bool f32 in new[] { false, true })
            {
                AudioFile.WriteWav(path, samples, 24000, f32);
                var back = AudioFile.ReadMono(path, 24000);
                Assert.Equal(samples.Length, back.Length);
                for (int i = 0; i < samples.Length; i++) Assert.True(Math.Abs(samples[i] - back[i]) < (f32 ? 1e-7 : 1e-4));
            }
        }
        finally
        {
            File.Delete(path);
        }
    }
}
