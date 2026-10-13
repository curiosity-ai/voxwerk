using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Voxwerk.Numerics;

/// <summary>bfloat16 helpers. A bf16 value is the upper 16 bits of an IEEE float32, so widening is a
/// zero-extend plus a 16-bit left shift - exact, which is why the checkpoint's bf16 weights are kept
/// in bf16 in memory (half the bandwidth of f32) without changing a single weight value.</summary>
public static unsafe class Bf16
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ToFloat(ushort v) => BitConverter.Int32BitsToSingle(v << 16);

    /// <summary>Round-to-nearest-even float32 → bf16 (matches torch's conversion).</summary>
    public static ushort FromFloat(float f)
    {
        uint bits = BitConverter.SingleToUInt32Bits(f);
        if ((bits & 0x7FFFFFFF) > 0x7F800000) return (ushort)((bits >> 16) | 0x40); // NaN
        uint rounding = 0x7FFF + ((bits >> 16) & 1);
        return (ushort)((bits + rounding) >> 16);
    }

    /// <summary>Widens <paramref name="n"/> bf16 values to float32.</summary>
    public static void ToFloat(ushort* src, float* dst, int n)
    {
        int i = 0;
        if (Avx512F.IsSupported)
        {
            for (; i <= n - 16; i += 16)
            {
                var w = Avx512F.ConvertToVector512UInt32(Vector256.Load(src + i));
                Vector512.Store(Avx512F.ShiftLeftLogical(w, 16).AsSingle(), dst + i);
            }
        }
        else if (Avx2.IsSupported)
        {
            for (; i <= n - 8; i += 8)
            {
                var w = Avx2.ConvertToVector256Int32(Vector128.Load(src + i));
                Vector256.Store(Avx2.ShiftLeftLogical(w, 16).AsSingle(), dst + i);
            }
        }
        for (; i < n; i++) dst[i] = ToFloat(src[i]);
    }

    /// <summary>Loads 16 bf16 values as a Vector512 of float32 (AVX-512).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Load512(ushort* p)
        => Avx512F.ShiftLeftLogical(Avx512F.ConvertToVector512UInt32(Vector256.Load(p)), 16).AsSingle();

    /// <summary>Loads 8 bf16 values as a Vector256 of float32 (AVX2).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Load256(ushort* p)
        => Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(Vector128.Load(p)), 16).AsSingle();
}
