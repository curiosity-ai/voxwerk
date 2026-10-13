using Voxwerk.IO;
using Voxwerk.Numerics;

namespace Voxwerk.Talker;

/// <summary>Storage precision of the talker / code predictor linear weights.</summary>
public enum WeightPrecision
{
    /// <summary>The checkpoint's native bfloat16 (exact; default).</summary>
    BFloat16,

    /// <summary>int8 with a per-output-channel scale: ~2× less memory traffic per decode step, slightly
    /// different numerics (opt-in speed mode).</summary>
    Int8,
}

/// <summary>Receives a layer's output hidden states (for tests and diagnostics).</summary>
public unsafe delegate void LayerObserverFn(int layer, float* hidden, int tokens);

/// <summary>Per-sequence key/value cache for a <see cref="DecoderStack"/>.</summary>
public sealed unsafe class KvCache : IDisposable
{
    private readonly NativeBuffer<float>[] _k, _v;

    public KvCache(int layers, int capacity, int rowSize)
    {
        Capacity = capacity;
        RowSize = rowSize;
        _k = new NativeBuffer<float>[layers];
        _v = new NativeBuffer<float>[layers];
        for (int i = 0; i < layers; i++)
        {
            _k[i] = new NativeBuffer<float>((long)capacity * rowSize, zero: false);
            _v[i] = new NativeBuffer<float>((long)capacity * rowSize, zero: false);
        }
    }

    public int Capacity { get; }
    public int RowSize { get; }
    /// <summary>Number of positions written so far.</summary>
    public int Length { get; set; }
    public float* K(int layer) => _k[layer].Ptr;
    public float* V(int layer) => _v[layer].Ptr;

    public void Dispose()
    {
        foreach (var b in _k) b.Dispose();
        foreach (var b in _v) b.Dispose();
    }
}

/// <summary>
/// A Qwen3 decoder stack as used by the talker and its code predictor: pre-norm layers with
/// RMS-normed q/k heads, RoPE, grouped-query attention and a SwiGLU MLP, followed by a final RMSNorm.
/// Weights stay in bf16, packed once; q/k/v and gate/up projections are fused into single GEMMs.
/// </summary>
public sealed unsafe class DecoderStack : IDisposable
{
    private sealed class Layer : IDisposable
    {
        public required PackedMatrix Qkv, O, GateUp, Down;
        public required float[] InNorm, PostNorm, QNorm, KNorm;

        public void Dispose()
        {
            Qkv.Dispose();
            O.Dispose();
            GateUp.Dispose();
            Down.Dispose();
        }
    }

    private readonly Layer[] _layers;
    private readonly float[] _norm;
    private readonly Rope _rope;
    private NativeBuffer<float>? _work;

    public DecoderStack(SafeTensorsFile st, string prefix, int layers, int hidden, int intermediate, int heads, int kvHeads, int headDim, float eps, double ropeTheta,
                        WeightPrecision precision = WeightPrecision.BFloat16)
    {
        Hidden = hidden;
        Intermediate = intermediate;
        Heads = heads;
        KvHeads = kvHeads;
        HeadDim = headDim;
        Eps = eps;
        _rope = new Rope(headDim, ropeTheta);
        _layers = new Layer[layers];
        for (int i = 0; i < layers; i++)
        {
            string p = $"{prefix}.layers.{i}.";
            using var q = st.ReadBf16(p + "self_attn.q_proj.weight");
            using var k = st.ReadBf16(p + "self_attn.k_proj.weight");
            using var v = st.ReadBf16(p + "self_attn.v_proj.weight");
            using var o = st.ReadBf16(p + "self_attn.o_proj.weight");
            using var g = st.ReadBf16(p + "mlp.gate_proj.weight");
            using var u = st.ReadBf16(p + "mlp.up_proj.weight");
            using var d = st.ReadBf16(p + "mlp.down_proj.weight");
            _layers[i] = new Layer
            {
                Qkv = Pack(precision, hidden, q, k, v),
                O = Pack(precision, heads * headDim, o),
                GateUp = Pack(precision, hidden, g, u),
                Down = Pack(precision, intermediate, d),
                InNorm = st.ReadFloat(p + "input_layernorm.weight"),
                PostNorm = st.ReadFloat(p + "post_attention_layernorm.weight"),
                QNorm = st.ReadFloat(p + "self_attn.q_norm.weight"),
                KNorm = st.ReadFloat(p + "self_attn.k_norm.weight"),
            };
        }
        _norm = st.ReadFloat(prefix + ".norm.weight");
    }

    internal static PackedMatrix Pack(WeightPrecision precision, int k, params NativeBuffer<ushort>[] parts)
        => precision == WeightPrecision.Int8 ? PackedMatrix.FromBf16LinearInt8(k, parts) : PackedMatrix.FromBf16Linear(k, parts);

    public int Hidden { get; }
    public int Intermediate { get; }
    public int Heads { get; }
    public int KvHeads { get; }
    public int HeadDim { get; }
    public float Eps { get; }
    public int LayerCount => _layers.Length;
    public long WeightBytes => _layers.Sum(l => l.Qkv.ByteSize + l.O.ByteSize + l.GateUp.ByteSize + l.Down.ByteSize);

    public KvCache CreateCache(int capacity) => new(_layers.Length, capacity, KvHeads * HeadDim);

    /// <summary>Optional per-layer hook (layer index, hidden states [T, hidden]) for parity tests.</summary>
    public LayerObserverFn? LayerObserver { get; set; }

    /// <summary>
    /// Runs <paramref name="tokens"/> new positions through the stack. <paramref name="x"/> [T, hidden] is the
    /// residual stream (updated in place); the final-normed hidden states are written to <paramref name="y"/>.
    /// </summary>
    public void Forward(float* x, int tokens, KvCache cache, float* y, Parallelism par)
    {
        int pos0 = cache.Length;
        if (pos0 + tokens > cache.Capacity) throw new InvalidOperationException("KV cache capacity exceeded.");
        int qDim = Heads * HeadDim, kvDim = KvHeads * HeadDim, qkvDim = qDim + 2 * kvDim;
        long need = (long)tokens * (Hidden + qkvDim + qDim + 2 * Intermediate);
        if (_work == null || _work.Length < need)
        {
            _work?.Dispose();
            _work = new NativeBuffer<float>(Math.Max(need, (long)4 * (Hidden + qkvDim + qDim + 2 * Intermediate)), zero: false);
        }
        float* h = _work.Ptr;
        float* qkv = h + (long)tokens * Hidden;
        float* attn = qkv + (long)tokens * qkvDim;
        float* gu = attn + (long)tokens * qDim;

        for (int li = 0; li < _layers.Length; li++)
        {
            var L = _layers[li];
            fixed (float* inNorm = L.InNorm, postNorm = L.PostNorm, qn = L.QNorm, kn = L.KNorm)
            {
                Ops.RmsNorm(x, h, tokens, Hidden, inNorm, Eps);
                Gemm.Multiply(h, Hidden, tokens, L.Qkv, qkv, qkvDim, par);
                float* kc = cache.K(li), vc = cache.V(li);
                for (int t = 0; t < tokens; t++)
                {
                    float* row = qkv + (long)t * qkvDim;
                    for (int hd = 0; hd < Heads; hd++) Ops.RmsNorm(row + hd * HeadDim, row + hd * HeadDim, 1, HeadDim, qn, Eps);
                    for (int hd = 0; hd < KvHeads; hd++) Ops.RmsNorm(row + qDim + hd * HeadDim, row + qDim + hd * HeadDim, 1, HeadDim, kn, Eps);
                    _rope.Apply(row, Heads, pos0 + t);
                    _rope.Apply(row + qDim, KvHeads, pos0 + t);
                    // Head-major cache: [kvHead][position][headDim].
                    for (int hd = 0; hd < KvHeads; hd++)
                    {
                        long dst = ((long)hd * cache.Capacity + pos0 + t) * HeadDim;
                        Buffer.MemoryCopy(row + qDim + hd * HeadDim, kc + dst, HeadDim * 4, HeadDim * 4);
                        Buffer.MemoryCopy(row + qDim + kvDim + hd * HeadDim, vc + dst, HeadDim * 4, HeadDim * 4);
                    }
                }
                Attention.CausalCached(qkv, qkvDim, tokens, pos0, kc, vc, cache.Capacity, Heads, KvHeads, HeadDim, attn, qDim, par);
                Gemm.Multiply(attn, qDim, tokens, L.O, x, Hidden, par, accumulate: true);

                Ops.RmsNorm(x, h, tokens, Hidden, postNorm, Eps);
                Gemm.Multiply(h, Hidden, tokens, L.GateUp, gu, 2 * Intermediate, par);
                for (int t = 0; t < tokens; t++)
                {
                    float* row = gu + (long)t * 2 * Intermediate;
                    Ops.SwiGlu(row, row + Intermediate, Intermediate);
                }
                Gemm.Multiply(gu, 2 * Intermediate, tokens, L.Down, x, Hidden, par, accumulate: true);
            }
            LayerObserver?.Invoke(li, x, tokens);
        }
        cache.Length = pos0 + tokens;
        fixed (float* nw = _norm) Ops.RmsNorm(x, y, tokens, Hidden, nw, Eps);
    }

    public void Dispose()
    {
        foreach (var l in _layers) l.Dispose();
        _work?.Dispose();
    }
}
