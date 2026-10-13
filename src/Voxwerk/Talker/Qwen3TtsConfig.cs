using System.Text.Json;

namespace Voxwerk.Talker;

/// <summary>The subset of <c>config.json</c> the runtime needs.</summary>
public sealed class Qwen3TtsConfig
{
    public int TtsBosTokenId { get; init; } = 151672;
    public int TtsEosTokenId { get; init; } = 151673;
    public int TtsPadTokenId { get; init; } = 151671;
    public string TtsModelType { get; init; } = "base";

    public int HiddenSize { get; init; } = 1024;
    public int IntermediateSize { get; init; } = 3072;
    public int NumLayers { get; init; } = 28;
    public int NumHeads { get; init; } = 16;
    public int NumKvHeads { get; init; } = 8;
    public int HeadDim { get; init; } = 128;
    public float RmsNormEps { get; init; } = 1e-6f;
    public double RopeTheta { get; init; } = 1_000_000;
    public int VocabSize { get; init; } = 3072;
    public int TextHiddenSize { get; init; } = 2048;
    public int TextVocabSize { get; init; } = 151936;
    public int NumCodeGroups { get; init; } = 16;

    public int CodecBosId { get; init; } = 2149;
    public int CodecEosId { get; init; } = 2150;
    public int CodecPadId { get; init; } = 2148;
    public int CodecThinkId { get; init; } = 2154;
    public int CodecNoThinkId { get; init; } = 2155;
    public int CodecThinkBosId { get; init; } = 2156;
    public int CodecThinkEosId { get; init; } = 2157;
    public IReadOnlyDictionary<string, int> LanguageIds { get; init; } = new Dictionary<string, int>();

    public int CpNumLayers { get; init; } = 5;
    public int CpHiddenSize { get; init; } = 1024;
    public int CpIntermediateSize { get; init; } = 3072;
    public int CpNumHeads { get; init; } = 16;
    public int CpNumKvHeads { get; init; } = 8;
    public int CpHeadDim { get; init; } = 128;
    public double CpRopeTheta { get; init; } = 1_000_000;
    public int CpVocabSize { get; init; } = 2048;

    public static Qwen3TtsConfig Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var r = doc.RootElement;
        var t = r.GetProperty("talker_config");
        var cp = t.GetProperty("code_predictor_config");
        var langs = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var p in t.GetProperty("codec_language_id").EnumerateObject()) langs[p.Name] = p.Value.GetInt32();
        return new Qwen3TtsConfig
        {
            TtsBosTokenId = r.GetProperty("tts_bos_token_id").GetInt32(),
            TtsEosTokenId = r.GetProperty("tts_eos_token_id").GetInt32(),
            TtsPadTokenId = r.GetProperty("tts_pad_token_id").GetInt32(),
            TtsModelType = r.TryGetProperty("tts_model_type", out var mt) ? mt.GetString()! : "base",
            HiddenSize = t.GetProperty("hidden_size").GetInt32(),
            IntermediateSize = t.GetProperty("intermediate_size").GetInt32(),
            NumLayers = t.GetProperty("num_hidden_layers").GetInt32(),
            NumHeads = t.GetProperty("num_attention_heads").GetInt32(),
            NumKvHeads = t.GetProperty("num_key_value_heads").GetInt32(),
            HeadDim = t.GetProperty("head_dim").GetInt32(),
            RmsNormEps = (float)t.GetProperty("rms_norm_eps").GetDouble(),
            RopeTheta = t.GetProperty("rope_theta").GetDouble(),
            VocabSize = t.GetProperty("vocab_size").GetInt32(),
            TextHiddenSize = t.GetProperty("text_hidden_size").GetInt32(),
            TextVocabSize = t.GetProperty("text_vocab_size").GetInt32(),
            NumCodeGroups = t.GetProperty("num_code_groups").GetInt32(),
            CodecBosId = t.GetProperty("codec_bos_id").GetInt32(),
            CodecEosId = t.GetProperty("codec_eos_token_id").GetInt32(),
            CodecPadId = t.GetProperty("codec_pad_id").GetInt32(),
            CodecThinkId = t.GetProperty("codec_think_id").GetInt32(),
            CodecNoThinkId = t.GetProperty("codec_nothink_id").GetInt32(),
            CodecThinkBosId = t.GetProperty("codec_think_bos_id").GetInt32(),
            CodecThinkEosId = t.GetProperty("codec_think_eos_id").GetInt32(),
            LanguageIds = langs,
            CpNumLayers = cp.GetProperty("num_hidden_layers").GetInt32(),
            CpHiddenSize = cp.GetProperty("hidden_size").GetInt32(),
            CpIntermediateSize = cp.GetProperty("intermediate_size").GetInt32(),
            CpNumHeads = cp.GetProperty("num_attention_heads").GetInt32(),
            CpNumKvHeads = cp.GetProperty("num_key_value_heads").GetInt32(),
            CpHeadDim = cp.GetProperty("head_dim").GetInt32(),
            CpRopeTheta = cp.GetProperty("rope_theta").GetDouble(),
            CpVocabSize = cp.GetProperty("vocab_size").GetInt32(),
        };
    }
}
