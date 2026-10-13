namespace Voxwerk.Talker;

/// <summary>
/// Everything the Base model needs to clone a voice: the speaker x-vector of the reference audio and, for
/// in-context-learning (ICL) mode, the reference transcript plus the reference audio's codec frames.
/// Build one with <c>Qwen3Tts.CreateVoiceClonePrompt</c> and reuse it for many generations.
/// </summary>
public sealed class VoiceClonePrompt
{
    /// <summary>Speaker embedding (x-vector) of the reference audio, length = talker hidden size.</summary>
    public required float[] SpeakerEmbedding { get; init; }

    /// <summary>Reference audio codec frames [frames][16] (ICL mode only).</summary>
    public int[][]? ReferenceCodes { get; init; }

    /// <summary>Transcript of the reference audio (ICL mode only).</summary>
    public string? ReferenceText { get; init; }

    /// <summary>True when only the x-vector is used (no transcript / reference codes).</summary>
    public bool XVectorOnly => ReferenceCodes == null || string.IsNullOrEmpty(ReferenceText);

    /// <summary>Serializes the prompt (speaker embedding, reference codes and text) to a compact binary form.</summary>
    public void Save(Stream stream)
    {
        using var w = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        w.Write(0x31505856); // "VXP1"
        w.Write(SpeakerEmbedding.Length);
        foreach (float f in SpeakerEmbedding) w.Write(f);
        w.Write(ReferenceText ?? "");
        int frames = ReferenceCodes?.Length ?? 0;
        w.Write(frames);
        for (int i = 0; i < frames; i++)
        {
            w.Write(ReferenceCodes![i].Length);
            foreach (int c in ReferenceCodes[i]) w.Write((short)c);
        }
    }

    /// <summary>Reads a prompt written by <see cref="Save"/>.</summary>
    public static VoiceClonePrompt Load(Stream stream)
    {
        using var r = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        if (r.ReadInt32() != 0x31505856) throw new InvalidDataException("Not a voice prompt file.");
        var emb = new float[r.ReadInt32()];
        for (int i = 0; i < emb.Length; i++) emb[i] = r.ReadSingle();
        string text = r.ReadString();
        int frames = r.ReadInt32();
        var codes = new int[frames][];
        for (int i = 0; i < frames; i++)
        {
            codes[i] = new int[r.ReadInt32()];
            for (int g = 0; g < codes[i].Length; g++) codes[i][g] = r.ReadInt16();
        }
        return new VoiceClonePrompt
        {
            SpeakerEmbedding = emb,
            ReferenceText = text.Length == 0 ? null : text,
            ReferenceCodes = frames == 0 ? null : codes,
        };
    }
}
