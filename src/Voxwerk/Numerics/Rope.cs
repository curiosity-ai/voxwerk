namespace Voxwerk.Numerics;

/// <summary>
/// Rotary position embedding in the "rotate_half" convention used by Qwen and Mimi:
/// <c>out[i] = x[i]·cos − x[i+d/2]·sin</c>, <c>out[i+d/2] = x[i+d/2]·cos + x[i]·sin</c>, with
/// angle <c>pos · θ^(−2i/d)</c>. Frequencies and angles are computed in float32 exactly as
/// <c>transformers</c>' default rope init does, so positions far into a sequence round the same way.
/// <para>The talker's multimodal RoPE (interleaved mrope, sections 24/20/20) degenerates to this one for
/// text/codec-only inputs because all three position streams are identical.</para>
/// </summary>
public sealed unsafe class Rope
{
    private readonly float[] _invFreq;
    private float[] _cos = [];
    private float[] _sin = [];
    private int _cached;
    private readonly object _lock = new();

    public Rope(int headDim, double theta)
    {
        HeadDim = headDim;
        _invFreq = new float[headDim / 2];
        for (int i = 0; i < headDim / 2; i++)
            _invFreq[i] = 1f / MathF.Pow((float)theta, (2 * i) / (float)headDim);
    }

    public int HeadDim { get; }

    private void Ensure(int positions)
    {
        if (positions <= _cached) return;
        lock (_lock)
        {
            if (positions <= _cached) return;
            int n = Math.Max(positions, _cached * 2);
            int half = HeadDim / 2;
            var cos = new float[n * half];
            var sin = new float[n * half];
            for (int p = 0; p < n; p++)
                for (int i = 0; i < half; i++)
                {
                    float f = p * _invFreq[i];
                    cos[p * half + i] = MathF.Cos(f);
                    sin[p * half + i] = MathF.Sin(f);
                }
            _cos = cos;
            _sin = sin;
            _cached = n;
        }
    }

    /// <summary>Rotates <paramref name="heads"/> consecutive head vectors of row <paramref name="x"/> at position <paramref name="pos"/>.</summary>
    public void Apply(float* x, int heads, int pos)
    {
        Ensure(pos + 1);
        int half = HeadDim / 2;
        fixed (float* cp = &_cos[pos * half], sp = &_sin[pos * half])
        {
            for (int h = 0; h < heads; h++)
            {
                float* v = x + h * HeadDim;
                for (int i = 0; i < half; i++)
                {
                    float x1 = v[i], x2 = v[i + half];
                    v[i] = x1 * cp[i] - x2 * sp[i];
                    v[i + half] = x2 * cp[i] + x1 * sp[i];
                }
            }
        }
    }
}
