namespace Voxwerk.Numerics;

/// <summary>
/// Recycles large native activation buffers. The vocoder runs at audio rate, so a single chunk touches
/// gigabytes of intermediate activations; fresh allocations would pay a kernel page fault (and zeroing)
/// per 4 KB page - measured at ~0.7 s per GB - while recycled buffers stay mapped.
/// </summary>
internal static class BufferPool
{
    private static readonly object Lock = new();
    private static readonly List<NativeBuffer<float>> Free = new();
    private static long _freeBytes;

    /// <summary>Upper bound on idle cached memory.</summary>
    public static long MaxCachedBytes { get; set; } = 3L << 30;

    public static NativeBuffer<float> Rent(long length)
    {
        lock (Lock)
        {
            int best = -1;
            for (int i = 0; i < Free.Count; i++)
            {
                long l = Free[i].Length;
                if (l >= length && l <= length * 2 + 4096 && (best < 0 || l < Free[best].Length)) best = i;
            }
            if (best >= 0)
            {
                var b = Free[best];
                Free.RemoveAt(best);
                _freeBytes -= b.Length * 4;
                return b;
            }
        }
        return new NativeBuffer<float>(length, zero: false);
    }

    public static void Return(NativeBuffer<float> buffer)
    {
        if (buffer.Length < (1 << 16))
        {
            buffer.Dispose();   // small buffers: not worth caching
            return;
        }
        lock (Lock)
        {
            Free.Add(buffer);
            _freeBytes += buffer.Length * 4;
            while (_freeBytes > MaxCachedBytes && Free.Count > 0)
            {
                int oldest = 0;
                _freeBytes -= Free[oldest].Length * 4;
                Free[oldest].Dispose();
                Free.RemoveAt(oldest);
            }
        }
    }

    /// <summary>Releases every cached buffer.</summary>
    public static void Trim()
    {
        lock (Lock)
        {
            foreach (var b in Free) b.Dispose();
            Free.Clear();
            _freeBytes = 0;
        }
    }
}
