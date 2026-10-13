using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace Voxwerk.Audio;

/// <summary>Audio file helpers built on NAudio (cross-platform managed WAV reading/writing and resampling).</summary>
public static class AudioFile
{
    /// <summary>Reads a WAV file and returns mono float samples resampled to <paramref name="sampleRate"/>.</summary>
    public static float[] ReadMono(string path, int sampleRate = 24000)
    {
        using var reader = new WaveFileReader(path);
        return ReadMono(reader.ToSampleProvider(), sampleRate);
    }

    /// <summary>Reads WAV bytes (e.g. a decoded base64 payload) as mono float at <paramref name="sampleRate"/>.</summary>
    public static float[] ReadMono(Stream wavStream, int sampleRate = 24000)
    {
        using var reader = new WaveFileReader(wavStream);
        return ReadMono(reader.ToSampleProvider(), sampleRate);
    }

    private static float[] ReadMono(ISampleProvider source, int sampleRate)
    {
        int channels = source.WaveFormat.Channels;
        var interleaved = new List<float>();
        var buf = new float[source.WaveFormat.SampleRate * channels];
        int read;
        while ((read = source.Read(buf.AsSpan())) > 0) interleaved.AddRange(buf.AsSpan(0, read));
        int frames = interleaved.Count / channels;
        var mono = new float[frames];
        for (int i = 0; i < frames; i++)
        {
            float s = 0;
            for (int c = 0; c < channels; c++) s += interleaved[i * channels + c];
            mono[i] = s / channels;
        }
        return Resample(mono, source.WaveFormat.SampleRate, sampleRate);
    }

    /// <summary>Resamples mono audio with NAudio's WDL resampler (no-op when the rates match).</summary>
    public static float[] Resample(float[] mono, int fromRate, int toRate)
    {
        if (fromRate == toRate) return mono;
        var src = new FloatArrayProvider(mono, fromRate);
        var rs = new WdlResamplingSampleProvider(src, toRate);
        var outList = new List<float>((int)((long)mono.Length * toRate / fromRate) + 1024);
        var buf = new float[8192];
        int n;
        while ((n = rs.Read(buf.AsSpan())) > 0) outList.AddRange(buf.AsSpan(0, n));
        return outList.ToArray();
    }

    /// <summary>Writes mono samples as a WAV file (16-bit PCM by default, or 32-bit float).</summary>
    public static void WriteWav(string path, ReadOnlySpan<float> samples, int sampleRate = 24000, bool float32 = false)
    {
        var format = float32 ? WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1) : new WaveFormat(sampleRate, 16, 1);
        using var writer = new WaveFileWriter(path, format);
        if (float32)
        {
            writer.WriteSamples(samples.ToArray(), 0, samples.Length);
        }
        else
        {
            var pcm = new short[samples.Length];
            for (int i = 0; i < pcm.Length; i++) pcm[i] = (short)Math.Round(Math.Clamp(samples[i], -1f, 1f) * short.MaxValue);
            writer.WriteSamples(pcm, 0, pcm.Length);
        }
    }

    private sealed class FloatArrayProvider(float[] data, int rate) : ISampleProvider
    {
        private int _pos;
        public WaveFormat WaveFormat { get; } = WaveFormat.CreateIeeeFloatWaveFormat(rate, 1);

        public int Read(Span<float> buffer)
        {
            int n = Math.Min(buffer.Length, data.Length - _pos);
            data.AsSpan(_pos, n).CopyTo(buffer);
            _pos += n;
            return n;
        }
    }
}
