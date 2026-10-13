using System.Text.Json.Serialization;

namespace Voxwerk.Voices;

/// <summary>Metadata of a packaged voice. The id has the form <c>&lt;language&gt;-&lt;country&gt;-&lt;name&gt;</c> in kebab-case,
/// e.g. <c>en-us-marigold-vale</c>.</summary>
public sealed record VoiceInfo
{
    /// <summary>Unique id, e.g. <c>en-gb-tobias-wren</c>.</summary>
    public required string Id { get; init; }

    /// <summary>Display name of the voice, e.g. "Tobias Wren".</summary>
    public required string Name { get; init; }

    /// <summary>Language id (ISO 639-1), e.g. <c>en</c>.</summary>
    public required string Language { get; init; }

    /// <summary>Country / region id used in the voice id, e.g. <c>us</c>, <c>gb</c>, <c>sct</c>, <c>ng</c>.</summary>
    public required string Country { get; init; }

    /// <summary>Package region, e.g. "American", "British", "African".</summary>
    public required string Region { get; init; }

    /// <summary>Accent / category within the region, e.g. "US Southern", "Received Pronunciation".</summary>
    public required string Category { get; init; }

    /// <summary><c>female</c>, <c>male</c> or <c>unknown</c>.</summary>
    public required string Gender { get; init; }

    /// <summary><c>young</c>, <c>middle-aged</c> or <c>old</c>.</summary>
    public string Age { get; init; } = "";

    /// <summary>Human-readable description of the voice.</summary>
    public required string Description { get; init; }

    /// <summary>Style / use-case tags, e.g. "narration", "calm", "deep".</summary>
    public IReadOnlyList<string> Tags { get; init; } = [];

    /// <summary>Exact transcript of the voice sample (used as the reference text for voice cloning).</summary>
    public required string SampleText { get; init; }

    /// <summary>Duration of the voice sample in seconds.</summary>
    public double SampleSeconds { get; init; }

    /// <summary>Text of the <see cref="VoiceSampleKind.Clone"/> sample (the same fixed sentence for every voice), or empty.</summary>
    public string CloneText { get; init; } = "";

    /// <summary>Duration of the clone sample in seconds (0 when the voice has none).</summary>
    public double CloneSeconds { get; init; }

    /// <summary>Name of the package that provides the voice (filled in on registration).</summary>
    [JsonIgnore]
    public string Package { get; init; } = "";

    /// <summary>True when every term of <paramref name="query"/> appears in the name, id, description, category, region or tags.</summary>
    public bool Matches(string query)
    {
        if (string.IsNullOrWhiteSpace(query)) return true;
        string hay = string.Join(' ', [Id, Name, Description, Category, Region, Country, Gender, Age, .. Tags]);
        return query.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                    .All(term => hay.Contains(term, StringComparison.OrdinalIgnoreCase));
    }
}

/// <summary>Which recording of a voice to fetch.</summary>
public enum VoiceSampleKind
{
    /// <summary>The voice introducing itself, generated from its description with the VoiceDesign model. This is the
    /// reference audio used when the voice is cloned (<see cref="VoiceInfo.SampleText"/> is its transcript).</summary>
    Design,

    /// <summary>The same voice re-created by voice cloning from the design sample (Base model), reading
    /// <see cref="VoiceInfo.CloneText"/> - a fixed sentence shared by all voices, for comparing the two approaches.</summary>
    Clone,
}

/// <summary>A decoded voice sample: 24 kHz mono audio, its kind and transcript.</summary>
public sealed record VoiceSample(VoiceInfo Info, VoiceSampleKind Kind, float[] Audio, int SampleRate)
{
    /// <summary>What the sample says.</summary>
    public string Text => Kind == VoiceSampleKind.Clone ? Info.CloneText : Info.SampleText;
}
