namespace Voxwerk.Numerics;

/// <summary>A channels-last sequence <c>[Time, Channels]</c> in native memory.</summary>
public sealed unsafe class Activation : IDisposable
{
    private readonly NativeBuffer<float> _buf;

    public Activation(int time, int channels, bool zero = false)
    {
        Time = time;
        Channels = channels;
        _buf = BufferPool.Rent(Math.Max(1L, (long)time * channels));
        if (zero) AsSpanLong().Clear();
    }


    public int Time { get; }
    public int Channels { get; }
    public long Length => (long)Time * Channels;
    public float* Ptr => _buf.Ptr;
    public float* Row(int t) => _buf.Ptr + (long)t * Channels;
    public Span<float> AsSpan() => new(Ptr, checked((int)Length));

    private Span<float> AsSpanLong() => new(_buf.Ptr, checked((int)Math.Max(1L, Length)));

    public float[] ToArray() => AsSpan().ToArray();

    public static Activation FromArray(float[] data, int time, int channels)
    {
        if (data.Length != (long)time * channels) throw new ArgumentException("size mismatch");
        var a = new Activation(time, channels);
        data.AsSpan().CopyTo(a.AsSpan());
        return a;
    }

    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        BufferPool.Return(_buf);
    }
}
