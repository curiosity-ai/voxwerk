namespace Voxwerk.Audio;

/// <summary>
/// Log-mel spectrogram matching Qwen3-TTS's <c>mel_spectrogram</c>: reflect padding of
/// <c>(n_fft − hop) / 2</c>, non-centered STFT with a periodic Hann window, magnitude
/// <c>sqrt(re² + im² + 1e-9)</c>, librosa's Slaney mel filterbank, and <c>log(max(x, 1e-5))</c>.
/// </summary>
public sealed class MelSpectrogram
{
    private readonly int _nFft, _hop, _nMels;
    private readonly float[] _window;
    private readonly float[] _filters;   // [nMels, nFft/2 + 1]
    private readonly double[] _cos, _sin;
    private readonly int[] _bitrev;

    public MelSpectrogram(int sampleRate = 24000, int nFft = 1024, int hop = 256, int nMels = 128, double fMin = 0, double fMax = 12000)
    {
        if ((nFft & (nFft - 1)) != 0) throw new ArgumentException("n_fft must be a power of two.");
        _nFft = nFft;
        _hop = hop;
        _nMels = nMels;
        _window = new float[nFft];
        for (int i = 0; i < nFft; i++) _window[i] = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * i / nFft));
        _filters = SlaneyFilterbank(sampleRate, nFft, nMels, fMin, fMax);
        _cos = new double[nFft / 2];
        _sin = new double[nFft / 2];
        for (int i = 0; i < nFft / 2; i++)
        {
            _cos[i] = Math.Cos(-2 * Math.PI * i / nFft);
            _sin[i] = Math.Sin(-2 * Math.PI * i / nFft);
        }
        int bits = (int)Math.Log2(nFft);
        _bitrev = new int[nFft];
        for (int i = 0; i < nFft; i++)
        {
            int r = 0;
            for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
            _bitrev[i] = r;
        }
    }

    public int Mels => _nMels;

    /// <summary>Returns [frames, nMels] (row-major).</summary>
    public float[] Compute(ReadOnlySpan<float> wav, out int frames)
    {
        int pad = (_nFft - _hop) / 2;
        int n = wav.Length;
        if (n <= pad) throw new ArgumentException("Audio is too short.");
        var y = new float[n + 2 * pad];
        for (int i = 0; i < pad; i++) y[i] = wav[pad - i];
        wav.CopyTo(y.AsSpan(pad));
        for (int i = 0; i < pad; i++) y[pad + n + i] = wav[n - 2 - i];
        frames = 1 + (y.Length - _nFft) / _hop;
        int bins = _nFft / 2 + 1;
        var mel = new float[frames * _nMels];
        var re = new double[_nFft];
        var im = new double[_nFft];
        var mag = new float[bins];
        for (int f = 0; f < frames; f++)
        {
            int off = f * _hop;
            for (int i = 0; i < _nFft; i++)
            {
                re[_bitrev[i]] = y[off + i] * _window[i];
                im[_bitrev[i]] = 0;
            }
            Fft(re, im);
            for (int k = 0; k < bins; k++) mag[k] = (float)Math.Sqrt(re[k] * re[k] + im[k] * im[k] + 1e-9);
            for (int m = 0; m < _nMels; m++)
            {
                float s = System.Numerics.Tensors.TensorPrimitives.Dot(new ReadOnlySpan<float>(_filters, m * bins, bins), mag);
                mel[f * _nMels + m] = MathF.Log(MathF.Max(s, 1e-5f));
            }
        }
        return mel;
    }

    /// <summary>In-place iterative radix-2 FFT (input already in bit-reversed order).</summary>
    private void Fft(double[] re, double[] im)
    {
        int n = re.Length;
        for (int size = 2; size <= n; size <<= 1)
        {
            int half = size >> 1, step = n / size;
            for (int start = 0; start < n; start += size)
            {
                for (int k = 0; k < half; k++)
                {
                    double wr = _cos[k * step], wi = _sin[k * step];
                    int a = start + k, b = a + half;
                    double tr = re[b] * wr - im[b] * wi;
                    double ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }

    private static double HzToMel(double f)
    {
        const double fSp = 200.0 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp;
        double logStep = Math.Log(6.4) / 27.0;
        return f >= minLogHz ? minLogMel + Math.Log(f / minLogHz) / logStep : f / fSp;
    }

    private static double MelToHz(double m)
    {
        const double fSp = 200.0 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp;
        double logStep = Math.Log(6.4) / 27.0;
        return m >= minLogMel ? minLogHz * Math.Exp(logStep * (m - minLogMel)) : fSp * m;
    }

    /// <summary><c>librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)</c> with htk=False, norm="slaney".</summary>
    internal static float[] SlaneyFilterbank(int sr, int nFft, int nMels, double fMin, double fMax)
    {
        int bins = nFft / 2 + 1;
        var fftFreqs = new double[bins];
        for (int i = 0; i < bins; i++) fftFreqs[i] = (double)i * sr / nFft;
        double mMin = HzToMel(fMin), mMax = HzToMel(fMax);
        var melF = new double[nMels + 2];
        for (int i = 0; i < nMels + 2; i++) melF[i] = MelToHz(mMin + (mMax - mMin) * i / (nMels + 1));
        var w = new float[nMels * bins];
        for (int m = 0; m < nMels; m++)
        {
            double lowDiff = melF[m + 1] - melF[m], highDiff = melF[m + 2] - melF[m + 1];
            double enorm = 2.0 / (melF[m + 2] - melF[m]);
            for (int k = 0; k < bins; k++)
            {
                double lower = -(melF[m] - fftFreqs[k]) / lowDiff;
                double upper = (melF[m + 2] - fftFreqs[k]) / highDiff;
                w[m * bins + k] = (float)(Math.Max(0, Math.Min(lower, upper)) * enorm);
            }
        }
        return w;
    }
}
