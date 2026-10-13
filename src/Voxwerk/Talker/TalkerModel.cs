using Voxwerk.IO;
using Voxwerk.Numerics;

namespace Voxwerk.Talker;

/// <summary>
/// The Qwen3-TTS talker: a 28-layer Qwen3 decoder that predicts the first codebook of each 12.5 Hz frame,
/// plus the 5-layer code predictor ("sub-talker") that fills in codebooks 1..15 of that frame. Text enters
/// through a 2048-wide text embedding and a SiLU MLP projection; codec tokens through codec embeddings.
/// </summary>
public sealed unsafe class TalkerModel : IDisposable
{
    private readonly NativeBuffer<ushort> _textEmbedding;   // [textVocab, textHidden] bf16
    private readonly PackedMatrix _textFc1, _textFc2;
    private readonly float[] _textFc1Bias, _textFc2Bias;
    private readonly NativeBuffer<ushort> _codecEmbedding;  // [vocab, hidden] bf16
    private readonly PackedMatrix _codecHead;
    private readonly NativeBuffer<ushort>[] _cpEmbedding;   // 15 × [cpVocab, hidden] bf16
    private readonly PackedMatrix[] _cpHead;                // 15 × (cpHidden → cpVocab)
    private readonly PackedMatrix? _toCp;                   // small_to_mtp_projection (hidden → cpHidden), 1.7B models
    private readonly float[]? _toCpBias;

    public TalkerModel(SafeTensorsFile st, Qwen3TtsConfig cfg, WeightPrecision precision = WeightPrecision.BFloat16)
    {
        Config = cfg;
        Precision = precision;
        _textEmbedding = st.ReadBf16("talker.model.text_embedding.weight");
        using (var w1 = st.ReadBf16("talker.text_projection.linear_fc1.weight"))
        using (var w2 = st.ReadBf16("talker.text_projection.linear_fc2.weight"))
        {
            _textFc1 = DecoderStack.Pack(precision, cfg.TextHiddenSize, w1);
            _textFc2 = DecoderStack.Pack(precision, cfg.TextHiddenSize, w2);
        }
        _textFc1Bias = st.ReadFloat("talker.text_projection.linear_fc1.bias");
        _textFc2Bias = st.ReadFloat("talker.text_projection.linear_fc2.bias");
        _codecEmbedding = st.ReadBf16("talker.model.codec_embedding.weight");
        using (var h = st.ReadBf16("talker.codec_head.weight")) _codecHead = DecoderStack.Pack(precision, cfg.HiddenSize, h);
        Model = new DecoderStack(st, "talker.model", cfg.NumLayers, cfg.HiddenSize, cfg.IntermediateSize, cfg.NumHeads,
            cfg.NumKvHeads, cfg.HeadDim, cfg.RmsNormEps, cfg.RopeTheta, precision);
        CodePredictor = new DecoderStack(st, "talker.code_predictor.model", cfg.CpNumLayers, cfg.CpHiddenSize, cfg.CpIntermediateSize,
            cfg.CpNumHeads, cfg.CpNumKvHeads, cfg.CpHeadDim, cfg.RmsNormEps, cfg.CpRopeTheta, precision);
        if (cfg.CpHiddenSize != cfg.HiddenSize)
        {
            using var w = st.ReadBf16("talker.code_predictor.small_to_mtp_projection.weight");
            _toCp = DecoderStack.Pack(precision, cfg.HiddenSize, w);
            _toCpBias = st.ReadFloat("talker.code_predictor.small_to_mtp_projection.bias");
        }
        int groups = cfg.NumCodeGroups - 1;
        _cpEmbedding = new NativeBuffer<ushort>[groups];
        _cpHead = new PackedMatrix[groups];
        for (int i = 0; i < groups; i++)
        {
            _cpEmbedding[i] = st.ReadBf16($"talker.code_predictor.model.codec_embedding.{i}.weight");
            using var h = st.ReadBf16($"talker.code_predictor.lm_head.{i}.weight");
            _cpHead[i] = DecoderStack.Pack(precision, cfg.CpHiddenSize, h);
        }
    }

    public Qwen3TtsConfig Config { get; }
    public WeightPrecision Precision { get; }
    public DecoderStack Model { get; }
    public DecoderStack CodePredictor { get; }
    public int Hidden => Config.HiddenSize;
    public int CpHidden => Config.CpHiddenSize;

    /// <summary>Maps talker-width rows to code-predictor width (<c>small_to_mtp_projection</c>; identity when equal).</summary>
    public void ToCodePredictor(float* x, int rows, float* y, Parallelism par)
    {
        if (_toCp == null)
        {
            Buffer.MemoryCopy(x, y, (long)rows * Hidden * 4, (long)rows * Hidden * 4);
            return;
        }
        fixed (float* b = _toCpBias) Gemm.Multiply(x, Hidden, rows, _toCp, y, CpHidden, par, b);
    }

    /// <summary><c>text_projection(text_embedding(ids))</c> → [ids.Length, hidden].</summary>
    public float[] ProjectText(ReadOnlySpan<int> ids, Parallelism par)
    {
        int n = ids.Length, th = Config.TextHiddenSize, h = Hidden;
        var result = new float[n * h];
        if (n == 0) return result;
        using var emb = new NativeBuffer<float>((long)n * th, zero: false);
        using var mid = new NativeBuffer<float>((long)n * th, zero: false);
        for (int i = 0; i < n; i++)
        {
            if ((uint)ids[i] >= (uint)Config.TextVocabSize) throw new ArgumentOutOfRangeException(nameof(ids), $"Token id {ids[i]} is outside the text vocabulary.");
            Bf16.ToFloat(_textEmbedding.Ptr + (long)ids[i] * th, emb.Ptr + (long)i * th, th);
        }
        fixed (float* b1 = _textFc1Bias, b2 = _textFc2Bias, r = result)
        {
            Gemm.Multiply(emb.Ptr, th, n, _textFc1, mid.Ptr, th, par, b1);
            Ops.Silu(mid.Ptr, (long)n * th);
            Gemm.Multiply(mid.Ptr, th, n, _textFc2, r, h, par, b2);
        }
        return result;
    }

    /// <summary>Adds (or writes) the talker codec embedding of <paramref name="id"/> to <paramref name="dst"/>.</summary>
    public void CodecEmbedding(int id, float* dst, bool add)
    {
        if ((uint)id >= (uint)Config.VocabSize) throw new ArgumentOutOfRangeException(nameof(id));
        AddRow(_codecEmbedding.Ptr + (long)id * Hidden, dst, Hidden, add);
    }

    /// <summary>Adds (or writes) the code predictor embedding of codebook <paramref name="group"/> (1..15).</summary>
    public void CodePredictorEmbedding(int group, int id, float* dst, bool add)
    {
        if ((uint)id >= (uint)Config.CpVocabSize) throw new ArgumentOutOfRangeException(nameof(id));
        AddRow(_cpEmbedding[group - 1].Ptr + (long)id * Hidden, dst, Hidden, add);
    }

    /// <summary>Sum of the 16 codebook embeddings of one frame (codebook 0 via the talker table).</summary>
    public void FrameEmbedding(ReadOnlySpan<int> codes, float* dst)
    {
        CodecEmbedding(codes[0], dst, add: false);
        for (int g = 1; g < codes.Length; g++) CodePredictorEmbedding(g, codes[g], dst, add: true);
    }

    private static void AddRow(ushort* src, float* dst, int n, bool add)
    {
        if (!add)
        {
            Bf16.ToFloat(src, dst, n);
            return;
        }
        float* tmp = stackalloc float[n];
        Bf16.ToFloat(src, tmp, n);
        for (int i = 0; i < n; i++) dst[i] += tmp[i];
    }

    /// <summary>Talker logits for one normed hidden row.</summary>
    public void CodecHead(float* hidden, float* logits, Parallelism par) => Gemm.Multiply(hidden, Hidden, 1, _codecHead, logits, Config.VocabSize, par);

    /// <summary>Code predictor head for codebook <paramref name="group"/> (1..15).</summary>
    public void CodePredictorHead(int group, float* hidden, float* logits, Parallelism par)
        => Gemm.Multiply(hidden, CpHidden, 1, _cpHead[group - 1], logits, Config.CpVocabSize, par);

    public void Dispose()
    {
        _textEmbedding.Dispose();
        _textFc1.Dispose();
        _textFc2.Dispose();
        _codecEmbedding.Dispose();
        _codecHead.Dispose();
        Model.Dispose();
        CodePredictor.Dispose();
        foreach (var e in _cpEmbedding) e.Dispose();
        foreach (var h in _cpHead) h.Dispose();
        _toCp?.Dispose();
    }
}
