namespace Voxwerk.IO;

/// <summary>Fetches the Qwen3-TTS checkpoint files from the Hugging Face Hub into a local directory.</summary>
public static class ModelDownloader
{
    public const string DefaultRepo = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";

    public static readonly string[] Files =
    [
        "config.json", "generation_config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors",
        "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors",
    ];

    /// <summary>Default cache location: <c>$VOXWERK_CACHE</c> or <c>~/.cache/voxwerk</c>.</summary>
    public static string DefaultCacheDir =>
        Environment.GetEnvironmentVariable("VOXWERK_CACHE") is { Length: > 0 } c
            ? c
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "voxwerk");

    /// <summary>Returns <paramref name="modelOrRepo"/> if it is an existing directory, otherwise downloads the
    /// repository's files (skipping ones already present) and returns the local directory.</summary>
    public static async Task<string> EnsureAsync(string modelOrRepo, string? cacheDir = null, IProgress<string>? progress = null, CancellationToken ct = default)
    {
        if (Directory.Exists(modelOrRepo)) return modelOrRepo;
        string dir = Path.Combine(cacheDir ?? DefaultCacheDir, modelOrRepo.Replace('/', Path.DirectorySeparatorChar));
        using var http = new HttpClient { Timeout = TimeSpan.FromHours(2) };
        string? token = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token)) http.DefaultRequestHeaders.Authorization = new("Bearer", token);
        foreach (var file in Files)
        {
            string dest = Path.Combine(dir, file.Replace('/', Path.DirectorySeparatorChar));
            if (File.Exists(dest)) continue;
            Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
            progress?.Report($"Downloading {modelOrRepo}/{file} ...");
            string url = $"https://huggingface.co/{modelOrRepo}/resolve/main/{file}";
            using var resp = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
            resp.EnsureSuccessStatusCode();
            string tmp = dest + ".partial";
            await using (var fs = File.Create(tmp))
                await resp.Content.CopyToAsync(fs, ct).ConfigureAwait(false);
            File.Move(tmp, dest, overwrite: true);
        }
        return dir;
    }
}
