using Voxwerk.IO;
using Voxwerk.Numerics;

namespace Voxwerk.Codec;

/// <summary>
/// The small causal transformers inside the speech tokenizer. Two flavours share the same skeleton
/// (pre-norm attention + MLP, each scaled by a learned LayerScale before the residual add, RoPE):
/// <list type="bullet">
/// <item>Mimi encoder: LayerNorm (with bias), GELU MLP, full causal attention.</item>
/// <item>Decoder pre-transformer: RMSNorm, SwiGLU MLP, sliding-window causal attention.</item>
/// </list>
/// </summary>
internal sealed unsafe class CodecTransformer : IDisposable
{
    private sealed class Layer : IDisposable
    {
        public required Linear Qkv, O, Up, Down;   // Up = fc1 (Mimi) or fused gate|up (decoder)
        public required float[] Norm1W, Norm2W, AttnScale, MlpScale;
        public float[]? Norm1B, Norm2B;

        public void Dispose()
        {
            Qkv.Dispose();
            O.Dispose();
            Up.Dispose();
            Down.Dispose();
        }
    }

    private readonly Layer[] _layers;
    private readonly bool _mimi;
    private readonly Rope _rope;

    public CodecTransformer(SafeTensorsFile st, string prefix, int layers, int hidden, int heads, int headDim, int intermediate,
                            float eps, double ropeTheta, int window, bool mimiStyle)
    {
        Hidden = hidden;
        Heads = heads;
        HeadDim = headDim;
        Intermediate = intermediate;
        Eps = eps;
        Window = window;
        _mimi = mimiStyle;
        _rope = new Rope(headDim, ropeTheta);
        _layers = new Layer[layers];
        for (int i = 0; i < layers; i++)
        {
            string p = $"{prefix}.layers.{i}.";
            _layers[i] = new Layer
            {
                Qkv = Linear.Load(st, p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"),
                O = Linear.Load(st, p + "self_attn.o_proj"),
                Up = mimiStyle ? Linear.Load(st, p + "mlp.fc1") : Linear.Load(st, p + "mlp.gate_proj", p + "mlp.up_proj"),
                Down = mimiStyle ? Linear.Load(st, p + "mlp.fc2") : Linear.Load(st, p + "mlp.down_proj"),
                Norm1W = st.ReadFloat(p + "input_layernorm.weight"),
                Norm2W = st.ReadFloat(p + "post_attention_layernorm.weight"),
                Norm1B = mimiStyle ? st.ReadFloat(p + "input_layernorm.bias") : null,
                Norm2B = mimiStyle ? st.ReadFloat(p + "post_attention_layernorm.bias") : null,
                AttnScale = st.ReadFloat(p + "self_attn_layer_scale.scale"),
                MlpScale = st.ReadFloat(p + "mlp_layer_scale.scale"),
            };
        }
    }

    public int Hidden { get; }
    public int Heads { get; }
    public int HeadDim { get; }
    public int Intermediate { get; }
    public float Eps { get; }
    public int Window { get; }

    private void Norm(float* x, float* y, int rows, float[] w, float[]? b)
    {
        fixed (float* wp = w, bp = b)
        {
            if (_mimi) Ops.LayerNorm(x, y, rows, Hidden, wp, bp, Eps);
            else Ops.RmsNorm(x, y, rows, Hidden, wp, Eps);
        }
    }

    /// <summary>Runs all layers over <paramref name="x"/> [T, hidden] in place (positions 0..T−1).</summary>
    public void Forward(Activation x, Parallelism par)
    {
        int T = x.Time, D = Heads * HeadDim, qkvDim = 3 * D;
        int upDim = _mimi ? Intermediate : 2 * Intermediate;
        using var h = new Activation(T, Hidden);
        using var qkv = new Activation(T, qkvDim);
        using var attn = new Activation(T, D);
        using var up = new Activation(T, upDim);
        using var tmp = new Activation(T, Hidden);
        foreach (var L in _layers)
        {
            Norm(x.Ptr, h.Ptr, T, L.Norm1W, L.Norm1B);
            L.Qkv.Forward(h.Ptr, Hidden, T, qkv.Ptr, qkvDim, par);
            for (int t = 0; t < T; t++)
            {
                _rope.Apply(qkv.Row(t), Heads, t);
                _rope.Apply(qkv.Row(t) + D, Heads, t);
            }
            Attention.Causal(qkv.Ptr, qkvDim, T, 0, qkv.Ptr + D, qkv.Ptr + 2 * D, qkvDim, Heads, Heads, HeadDim, Window, attn.Ptr, D, par);
            L.O.Forward(attn.Ptr, D, T, tmp.Ptr, Hidden, par);
            ScaledAdd(x, tmp, L.AttnScale);

            Norm(x.Ptr, h.Ptr, T, L.Norm2W, L.Norm2B);
            L.Up.Forward(h.Ptr, Hidden, T, up.Ptr, upDim, par);
            if (_mimi)
            {
                Ops.Gelu(up.Ptr, up.Length);
            }
            else
            {
                for (int t = 0; t < T; t++) Ops.SwiGlu(up.Row(t), up.Row(t) + Intermediate, Intermediate);
            }
            L.Down.Forward(up.Ptr, upDim, T, tmp.Ptr, Hidden, par);
            ScaledAdd(x, tmp, L.MlpScale);
        }
    }

    private static void ScaledAdd(Activation x, Activation d, float[] scale)
    {
        for (int t = 0; t < x.Time; t++)
            System.Numerics.Tensors.TensorPrimitives.FusedMultiplyAdd(
                new ReadOnlySpan<float>(d.Row(t), x.Channels), scale, new ReadOnlySpan<float>(x.Row(t), x.Channels), new Span<float>(x.Row(t), x.Channels));
    }

    public void Dispose()
    {
        foreach (var l in _layers) l.Dispose();
    }
}
