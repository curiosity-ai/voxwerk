using System.Buffers.Binary;
using System.Text.Json;
using Microsoft.Win32.SafeHandles;
using Voxwerk.Numerics;

namespace Voxwerk.IO;

/// <summary>
/// Reader for the Hugging Face <c>safetensors</c> format: an 8-byte little-endian header length, a JSON
/// header mapping tensor names to <c>dtype</c> / <c>shape</c> / <c>data_offsets</c>, then the raw data.
/// Tensors are read on demand straight into aligned native memory with positional reads, so loading a
/// 1.8 GB checkpoint never materializes it twice.
/// </summary>
public sealed unsafe class SafeTensorsFile : IDisposable
{
    private readonly SafeFileHandle _handle;
    private readonly long _dataStart;
    private readonly Dictionary<string, TensorInfo> _tensors;

    public readonly record struct TensorInfo(string Name, string DType, int[] Shape, long Begin, long End)
    {
        public long ElementCount
        {
            get
            {
                long n = 1;
                foreach (int d in Shape) n *= d;
                return n;
            }
        }
    }

    public SafeTensorsFile(string path)
    {
        Path = path;
        _handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.RandomAccess);
        long fileLength = RandomAccess.GetLength(_handle);
        Span<byte> lenBytes = stackalloc byte[8];
        if (fileLength < 8 || RandomAccess.Read(_handle, lenBytes, 0) != 8)
            throw new InvalidDataException($"'{path}' is not a safetensors file.");
        ulong headerLen = BinaryPrimitives.ReadUInt64LittleEndian(lenBytes);
        if (headerLen == 0 || headerLen > 100_000_000 || 8 + (long)headerLen > fileLength)
            throw new InvalidDataException($"'{path}' has an invalid safetensors header length ({headerLen}).");
        var header = new byte[headerLen];
        RandomAccess.Read(_handle, header, 8);
        _dataStart = 8 + (long)headerLen;
        _tensors = new Dictionary<string, TensorInfo>(StringComparer.Ordinal);
        using var doc = JsonDocument.Parse(header);
        foreach (var prop in doc.RootElement.EnumerateObject())
        {
            if (prop.NameEquals("__metadata__")) continue;
            var e = prop.Value;
            var shapeEl = e.GetProperty("shape");
            var shape = new int[shapeEl.GetArrayLength()];
            int i = 0;
            foreach (var d in shapeEl.EnumerateArray()) shape[i++] = d.GetInt32();
            var off = e.GetProperty("data_offsets");
            var info = new TensorInfo(prop.Name, e.GetProperty("dtype").GetString()!, shape, off[0].GetInt64(), off[1].GetInt64());
            if (_dataStart + info.End > fileLength)
                throw new InvalidDataException($"'{path}' is truncated (tensor '{prop.Name}' ends past the end of the file). Delete it and download again.");
            _tensors[prop.Name] = info;
        }
    }

    public string Path { get; }

    public IReadOnlyCollection<string> Names => _tensors.Keys;

    public bool Contains(string name) => _tensors.ContainsKey(name);

    public TensorInfo Info(string name)
        => _tensors.TryGetValue(name, out var t) ? t : throw new KeyNotFoundException($"Tensor '{name}' not found in '{Path}'.");

    private void ReadBytes(in TensorInfo t, byte* dst)
    {
        long total = t.End - t.Begin;
        long done = 0;
        while (done < total)
        {
            int chunk = (int)Math.Min(1 << 30, total - done);
            int read = RandomAccess.Read(_handle, new Span<byte>(dst + done, chunk), _dataStart + t.Begin + done);
            if (read <= 0) throw new EndOfStreamException($"Unexpected end of file reading '{t.Name}'.");
            done += read;
        }
    }

    /// <summary>Reads a BF16 tensor without conversion.</summary>
    public NativeBuffer<ushort> ReadBf16(string name)
    {
        var t = Info(name);
        if (t.DType != "BF16") throw new NotSupportedException($"'{name}' is {t.DType}, expected BF16.");
        var buf = new NativeBuffer<ushort>(t.ElementCount, zero: false);
        ReadBytes(t, (byte*)buf.Ptr);
        return buf;
    }

    /// <summary>Reads any supported tensor widened to float32.</summary>
    public float[] ReadFloat(string name)
    {
        var t = Info(name);
        var result = new float[t.ElementCount];
        fixed (float* dst = result) ReadFloatInto(t, dst);
        return result;
    }

    public NativeBuffer<float> ReadFloatNative(string name)
    {
        var t = Info(name);
        var buf = new NativeBuffer<float>(t.ElementCount, zero: false);
        ReadFloatInto(t, buf.Ptr);
        return buf;
    }

    private void ReadFloatInto(in TensorInfo t, float* dst)
    {
        long n = t.ElementCount;
        switch (t.DType)
        {
            case "F32":
                ReadBytes(t, (byte*)dst);
                if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException();
                break;
            case "BF16":
            {
                using var tmp = new NativeBuffer<ushort>(n, zero: false);
                ReadBytes(t, (byte*)tmp.Ptr);
                for (long i = 0; i < n; i += 1 << 20)
                    Bf16.ToFloat(tmp.Ptr + i, dst + i, (int)Math.Min(1 << 20, n - i));
                break;
            }
            case "F16":
            {
                using var tmp = new NativeBuffer<ushort>(n, zero: false);
                ReadBytes(t, (byte*)tmp.Ptr);
                for (long i = 0; i < n; i++) dst[i] = (float)BitConverter.UInt16BitsToHalf(tmp.Ptr[i]);
                break;
            }
            default:
                throw new NotSupportedException($"Unsupported dtype '{t.DType}' for tensor '{t.Name}'.");
        }
    }

    public void Dispose() => _handle.Dispose();
}
