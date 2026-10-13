namespace Voxwerk.Talker;

/// <summary>
/// Everything the Base model needs to clone a voice: the speaker x-vector of the reference audio and, for
/// in-context-learning (ICL) mode, the reference transcript plus the reference audio's codec frames.
/// Build one with <see cref="Qwen3Tts.CreateVoiceClonePrompt"/> and reuse it for many generations.
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
}
