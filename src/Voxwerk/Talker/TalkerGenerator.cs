using Voxwerk.Numerics;
using Voxwerk.Text;

namespace Voxwerk.Talker;

/// <summary>Generation settings (defaults follow the model's <c>generation_config.json</c>).</summary>
public sealed record GenerationOptions
{
    /// <summary>Upper bound on generated 12.5 Hz frames (2048 ≈ 164 s).</summary>
    public int MaxNewTokens { get; init; } = 2048;
    public float RepetitionPenalty { get; init; } = 1.05f;
    public SamplingOptions Talker { get; init; } = new();
    public SamplingOptions SubTalker { get; init; } = new();
    /// <summary>Feed the whole text in the prompt instead of streaming it alongside the generated codes.
    /// <c>null</c> uses the reference default for the mode: streaming for voice cloning, non-streaming for
    /// voice design and custom voices.</summary>
    public bool? NonStreamingMode { get; init; }
    public int? Seed { get; init; }

    /// <summary>Deterministic argmax decoding for both stages (used by the parity tests).</summary>
    public static GenerationOptions Greedy { get; } = new() { Talker = SamplingOptions.Greedy, SubTalker = SamplingOptions.Greedy };
}

/// <summary>Builds the talker prompt and runs the two-level autoregressive loop that produces codec frames.</summary>
public sealed unsafe class TalkerGenerator
{
    private readonly TalkerModel _m;
    private readonly Qwen2Tokenizer _tok;
    private readonly Qwen3TtsConfig _c;

    public TalkerGenerator(TalkerModel model, Qwen2Tokenizer tokenizer)
    {
        _m = model;
        _tok = tokenizer;
        _c = model.Config;
    }

    /// <summary>Optional hook receiving named intermediate tensors (prefill inputs, logits) for tests.</summary>
    public Action<string, float[]>? Trace { get; set; }

    public sealed class Prompt
    {
        public required float[] Prefill { get; init; }       // [length, hidden]
        public required int Length { get; init; }
        public required float[] Trailing { get; init; }      // [trailingLength, hidden] text still to be fed
        public required int TrailingLength { get; init; }
        public required float[] TtsPad { get; init; }        // [hidden]
    }

    public static string AssistantText(string text) => $"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n";
    public static string RefText(string text) => $"<|im_start|>assistant\n{text}<|im_end|>\n";
    public static string InstructText(string text) => $"<|im_start|>user\n{text}<|im_end|>\n";

    /// <summary>Voice-clone prompt (Base models).</summary>
    public Prompt BuildPrompt(string text, string language, VoiceClonePrompt voice, bool nonStreaming, Parallelism par)
        => BuildPrompt(text, language, voice.SpeakerEmbedding, null, null,
                       voice.XVectorOnly ? null : voice.ReferenceCodes, voice.XVectorOnly ? null : voice.ReferenceText, nonStreaming, par);

    /// <summary>
    /// Builds the talker input exactly like <c>Qwen3TTSForConditionalGeneration.generate</c>:
    /// [instruction] + role + codec prefix (think / language tags, optional speaker) + text, with either
    /// ICL reference codes (voice clone) or the text streamed / laid out in full.
    /// </summary>
    /// <param name="speakerEmbedding">x-vector of a cloned voice, or null.</param>
    /// <param name="speaker">Named CustomVoice speaker (its codec embedding becomes the speaker slot), or null.</param>
    /// <param name="instruct">Natural-language voice / style instruction, or null.</param>
    public Prompt BuildPrompt(string text, string language, float[]? speakerEmbedding, string? speaker, string? instruct,
                              int[][]? referenceCodes, string? referenceText, bool nonStreaming, Parallelism par)
    {
        int H = _m.Hidden;
        int[] ids = _tok.Encode(AssistantText(text));
        if (ids.Length < 9) throw new ArgumentException("Text is empty.", nameof(text));
        language = string.IsNullOrEmpty(language) ? "auto" : language;

        var special = _m.ProjectText([_c.TtsBosTokenId, _c.TtsEosTokenId, _c.TtsPadTokenId], par);
        float[] ttsBos = special[..H], ttsEos = special[H..(2 * H)], ttsPad = special[(2 * H)..];

        float[]? speakerRow = speakerEmbedding;
        if (speaker != null)
        {
            if (!_c.SpeakerIds.TryGetValue(speaker, out int spkId))
                throw new ArgumentException($"Unknown speaker '{speaker}'. Available: {string.Join(", ", _c.SpeakerIds.Keys)}.", nameof(speaker));
            speakerRow = new float[H];
            fixed (float* sp = speakerRow) _m.CodecEmbedding(spkId, sp, add: false);
        }
        if (speakerRow != null && speakerRow.Length != H)
            throw new ArgumentException($"Speaker embedding has {speakerRow.Length} values, the model expects {H}.");

        // Language tag; dialect speakers replace Chinese / auto with their dialect.
        int? langId = null;
        if (!language.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            if (language.Contains("dialect", StringComparison.OrdinalIgnoreCase) || !_c.LanguageIds.TryGetValue(language, out int lid))
                throw new ArgumentException($"Unsupported language '{language}'. Supported: auto, {string.Join(", ", _c.LanguageIds.Keys.Where(k => !k.Contains("dialect")))}.", nameof(language));
            langId = lid;
        }
        if (speaker != null && (language.Equals("auto", StringComparison.OrdinalIgnoreCase) || language.Equals("chinese", StringComparison.OrdinalIgnoreCase))
            && _c.SpeakerDialects.TryGetValue(speaker, out var dialect))
            langId = _c.LanguageIds[dialect];

        var codecIds = new List<int>();
        if (langId is int l) codecIds.AddRange([_c.CodecThinkId, _c.CodecThinkBosId, l, _c.CodecThinkEosId]);
        else codecIds.AddRange([_c.CodecNoThinkId, _c.CodecThinkBosId, _c.CodecThinkEosId]);
        int nTags = codecIds.Count;
        int spkSlots = speakerRow != null ? 1 : 0;
        int codecLen = nTags + spkSlots + 2;
        var codec = new float[codecLen * H];
        fixed (float* cp = codec)
        {
            for (int i = 0; i < nTags; i++) _m.CodecEmbedding(codecIds[i], cp + i * H, add: false);
            speakerRow?.AsSpan().CopyTo(codec.AsSpan(nTags * H, H));
            _m.CodecEmbedding(_c.CodecPadId, cp + (nTags + spkSlots) * H, add: false);
            _m.CodecEmbedding(_c.CodecBosId, cp + (nTags + spkSlots + 1) * H, add: false);
        }

        var rows = new List<float[]>();
        if (!string.IsNullOrEmpty(instruct))
        {
            var ins = _m.ProjectText(_tok.Encode(InstructText(instruct)), par);
            for (int i = 0; i < ins.Length / H; i++) rows.Add(ins[(i * H)..((i + 1) * H)]);
        }
        // <|im_start|>assistant\n
        var role = _m.ProjectText(ids.AsSpan(0, 3), par);
        for (int i = 0; i < 3; i++) rows.Add(role[(i * H)..((i + 1) * H)]);
        // tts_pad × (codecLen − 2), tts_bos  +  codec[:-1]
        for (int i = 0; i < codecLen - 1; i++)
        {
            var r = (i < codecLen - 2 ? ttsPad : ttsBos).ToArray();
            for (int j = 0; j < H; j++) r[j] += codec[i * H + j];
            rows.Add(r);
        }
        float[] codecBos = codec[((codecLen - 1) * H)..];

        int[] textIds = ids[3..^5];
        float[] trailing;
        int trailingLen;
        if (referenceCodes != null && !string.IsNullOrEmpty(referenceText))
        {
            int[] refIds = _tok.Encode(RefText(referenceText));
            int[] icl = [.. refIds[3..^2], .. textIds];
            var textEmb = Concat(_m.ProjectText(icl, par), ttsEos);
            int textLen = textEmb.Length / H;
            var refCodes = referenceCodes;
            int cLen = refCodes.Length + 1;
            var codecEmb = new float[cLen * H];
            fixed (float* ce = codecEmb)
            {
                _m.CodecEmbedding(_c.CodecBosId, ce, add: false);
                for (int f = 0; f < refCodes.Length; f++) _m.FrameEmbedding(refCodes[f], ce + (f + 1) * H);
            }
            if (nonStreaming)
            {
                var pad = new float[H];
                fixed (float* pp = pad) _m.CodecEmbedding(_c.CodecPadId, pp, add: false);
                for (int i = 0; i < textLen; i++) rows.Add(Add(textEmb.AsSpan(i * H, H), pad));
                for (int i = 0; i < cLen; i++) rows.Add(Add(codecEmb.AsSpan(i * H, H), ttsPad));
                trailing = ttsPad;
                trailingLen = 1;
            }
            else if (textLen > cLen)
            {
                for (int i = 0; i < cLen; i++) rows.Add(Add(textEmb.AsSpan(i * H, H), codecEmb.AsSpan(i * H, H)));
                trailing = textEmb[(cLen * H)..];
                trailingLen = textLen - cLen;
            }
            else
            {
                for (int i = 0; i < cLen; i++)
                    rows.Add(Add(i < textLen ? textEmb.AsSpan(i * H, H) : ttsPad, codecEmb.AsSpan(i * H, H)));
                trailing = ttsPad;
                trailingLen = 1;
            }
        }
        else if (nonStreaming)
        {
            var textEmb = Concat(_m.ProjectText(textIds, par), ttsEos);
            var pad = new float[H];
            fixed (float* pp = pad) _m.CodecEmbedding(_c.CodecPadId, pp, add: false);
            for (int i = 0; i < textEmb.Length / H; i++) rows.Add(Add(textEmb.AsSpan(i * H, H), pad));
            rows.Add(Add(ttsPad, codecBos));
            trailing = ttsPad;
            trailingLen = 1;
        }
        else
        {
            var first = _m.ProjectText(textIds.AsSpan(0, 1), par);
            rows.Add(Add(first, codecBos));
            trailing = Concat(_m.ProjectText(textIds.AsSpan(1), par), ttsEos);
            trailingLen = trailing.Length / H;
        }

        var prefill = new float[rows.Count * H];
        for (int i = 0; i < rows.Count; i++) rows[i].CopyTo(prefill, i * H);
        return new Prompt { Prefill = prefill, Length = rows.Count, Trailing = trailing, TrailingLength = trailingLen, TtsPad = ttsPad };
    }

    private static float[] Concat(float[] a, float[] b) => [.. a, .. b];

    private static float[] Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var r = new float[a.Length];
        for (int i = 0; i < r.Length; i++) r[i] = a[i] + b[i];
        return r;
    }

    /// <summary>Runs the talker + code predictor and returns codec frames [frames][16].</summary>
    public List<int[]> Generate(Prompt prompt, GenerationOptions opts, Parallelism par, CancellationToken ct = default,
                                Action<int>? onFrame = null)
    {
        int H = _m.Hidden, V = _c.VocabSize, G = _c.NumCodeGroups;
        var sampler = new Sampler(opts.Seed);
        var subSampler = new Sampler(opts.Seed is int s ? s + 1 : null);
        Trace?.Invoke("talker_prefill_in", prompt.Prefill);

        using var cache = _m.Model.CreateCache(prompt.Length + opts.MaxNewTokens + 1);
        using var cpCache = _m.CodePredictor.CreateCache(G + 1);
        int Hc = _m.CpHidden;
        using var buf = new NativeBuffer<float>((long)prompt.Length * H * 2 + 4 * H + 4 * Hc + V + 4096);
        float* x = buf.Ptr, y = x + (long)prompt.Length * H;
        float* last = y + (long)prompt.Length * H;   // talker hidden (normed) of the latest position
        float* step = last + H;                      // next talker input
        float* cpX = step + H;                       // [2, H] code predictor inputs at talker width
        float* cpIn = cpX + 2 * H;                   // [2, Hc] projected inputs
        float* cpY = cpIn + 2 * Hc;                  // [2, Hc]
        float* logits = cpY + 2 * Hc;                // [V]

        fixed (float* pp = prompt.Prefill) Buffer.MemoryCopy(pp, x, (long)prompt.Length * H * 4, (long)prompt.Length * H * 4);
        _m.Model.Forward(x, prompt.Length, cache, y, par);
        Buffer.MemoryCopy(y + (long)(prompt.Length - 1) * H, last, H * 4, H * 4);
        _m.CodecHead(last, logits, par);

        var history = new List<int>();
        var frames = new List<int[]>();
        var cpLogits = new float[_c.CpVocabSize];
        for (int t = 0; t < opts.MaxNewTokens; t++)
        {
            ct.ThrowIfCancellationRequested();
            var lg = new Span<float>(logits, V);
            Trace?.Invoke("talker_logits", lg.ToArray());
            Sampler.ApplyRepetitionPenalty(lg, history, opts.RepetitionPenalty);
            for (int i = V - 1024; i < V; i++) if (i != _c.CodecEosId) lg[i] = float.NegativeInfinity;
            if (t < 2) lg[_c.CodecEosId] = float.NegativeInfinity;   // min_new_tokens = 2
            int code0 = sampler.Select(lg, opts.Talker);
            history.Add(code0);
            if (code0 == _c.CodecEosId || t == opts.MaxNewTokens - 1) break;

            // Code predictor: [talker hidden, embed(code0)] → codebooks 1..15.
            var frame = new int[G];
            frame[0] = code0;
            cpCache.Length = 0;
            Buffer.MemoryCopy(last, cpX, H * 4, H * 4);
            _m.CodecEmbedding(code0, cpX + H, add: false);
            _m.ToCodePredictor(cpX, 2, cpIn, par);
            _m.CodePredictor.Forward(cpIn, 2, cpCache, cpY, par);
            float* cpLast = cpY + Hc;
            for (int g = 1; g < G; g++)
            {
                fixed (float* cl = cpLogits)
                {
                    _m.CodePredictorHead(g, cpLast, cl, par);
                    Trace?.Invoke("cp_logits", (float[])cpLogits.Clone());
                    frame[g] = subSampler.Select(cpLogits, opts.SubTalker);
                }
                if (g == G - 1) break;
                _m.CodePredictorEmbedding(g, frame[g], cpX, add: false);
                _m.ToCodePredictor(cpX, 1, cpIn, par);
                _m.CodePredictor.Forward(cpIn, 1, cpCache, cpY, par);
                cpLast = cpY;
            }
            frames.Add(frame);
            onFrame?.Invoke(frames.Count);

            // Next talker input: Σ codebook embeddings + next text embedding (or tts_pad once the text is consumed).
            _m.FrameEmbedding(frame, step);
            float[] text = t < prompt.TrailingLength ? prompt.Trailing : prompt.TtsPad;
            int off = t < prompt.TrailingLength ? t * H : 0;
            for (int i = 0; i < H; i++) step[i] += text[off + i];
            Trace?.Invoke("talker_step_in", new Span<float>(step, H).ToArray());
            _m.Model.Forward(step, 1, cache, last, par);
            _m.CodecHead(last, logits, par);
        }
        return frames;
    }
}
