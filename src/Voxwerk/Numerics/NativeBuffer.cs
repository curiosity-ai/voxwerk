using System.Runtime.InteropServices;

namespace Voxwerk.Numerics;

/// <summary>A 64-byte aligned block of unmanaged memory. Weights and large activations live here so the
/// SIMD kernels can address them with raw pointers (no pinning, no bounds checks, no GC moves).</summary>
public sealed unsafe class NativeBuffer<T> : IDisposable where T : unmanaged
{
    private T* _ptr;

    public NativeBuffer(long length, bool zero = true)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        Length = length;
        nuint bytes = (nuint)Math.Max(1, length * sizeof(T));
        _ptr = (T*)NativeMemory.AlignedAlloc(bytes, 64);
        if (zero) NativeMemory.Clear(_ptr, bytes);
        GC.AddMemoryPressure((long)bytes);
    }

    public long Length { get; }

    public T* Ptr
    {
        get
        {
            ObjectDisposedException.ThrowIf(_ptr == null, this);
            return _ptr;
        }
    }

    public Span<T> AsSpan() => new(Ptr, checked((int)Length));

    public Span<T> AsSpan(long start, int length)
    {
        if (start < 0 || start + length > Length) throw new ArgumentOutOfRangeException(nameof(start));
        return new Span<T>(Ptr + start, length);
    }

    public void Clear() => NativeMemory.Clear(Ptr, (nuint)(Length * sizeof(T)));

    public void Dispose()
    {
        if (_ptr != null)
        {
            NativeMemory.AlignedFree(_ptr);
            _ptr = null;
            GC.RemoveMemoryPressure(Math.Max(1, Length * sizeof(T)));
        }
        GC.SuppressFinalize(this);
    }

    ~NativeBuffer()
    {
        if (_ptr != null) NativeMemory.AlignedFree(_ptr);
    }
}
