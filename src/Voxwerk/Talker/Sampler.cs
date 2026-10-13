namespace Voxwerk.Talker;

/// <summary>Decoding settings for one stage (talker or code predictor), mirroring Hugging Face <c>generate</c>.</summary>
public sealed record SamplingOptions
{
    /// <summary>Sample (true) or take the argmax (false).</summary>
    public bool DoSample { get; init; } = true;
    public int TopK { get; init; } = 50;
    public float TopP { get; init; } = 1.0f;
    public float Temperature { get; init; } = 0.9f;

    public static SamplingOptions Greedy { get; } = new() { DoSample = false };
}

/// <summary>Logit processing + token selection with the semantics of the HF logits warpers
/// (temperature → top-k → top-p → multinomial).</summary>
public sealed class Sampler
{
    private readonly Random _rng;

    public Sampler(int? seed) => _rng = seed is int s ? new Random(s) : new Random();

    /// <summary>HF <c>RepetitionPenaltyLogitsProcessor</c>: every previously generated id is divided (positive
    /// logit) or multiplied (negative logit) by <paramref name="penalty"/>.</summary>
    public static void ApplyRepetitionPenalty(Span<float> logits, IEnumerable<int> history, float penalty)
    {
        if (penalty == 1f) return;
        foreach (int id in history.Distinct())
        {
            float s = logits[id];
            logits[id] = s < 0 ? s * penalty : s / penalty;
        }
    }

    public int Select(Span<float> logits, SamplingOptions opts)
    {
        if (!opts.DoSample) return ArgMax(logits);
        int n = logits.Length;
        float invT = 1f / opts.Temperature;
        for (int i = 0; i < n; i++) logits[i] *= invT;

        if (opts.TopK > 0 && opts.TopK < n)
        {
            // Keep every logit >= the k-th largest (ties kept, like torch.topk-based filtering).
            var sorted = logits.ToArray();
            Array.Sort(sorted);
            float kth = sorted[n - opts.TopK];
            for (int i = 0; i < n; i++) if (logits[i] < kth) logits[i] = float.NegativeInfinity;
        }

        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) max = Math.Max(max, logits[i]);
        var probs = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            probs[i] = float.IsNegativeInfinity(logits[i]) ? 0 : Math.Exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < n; i++) probs[i] /= sum;

        if (opts.TopP < 1f)
        {
            // HF TopP: sort ascending, drop the low tail whose cumulative mass is <= 1 - top_p (always keep the top token).
            var idx = Enumerable.Range(0, n).OrderBy(i => probs[i]).ToArray();
            double cum = 0;
            for (int j = 0; j < n - 1; j++)
            {
                cum += probs[idx[j]];
                if (cum <= 1 - opts.TopP) probs[idx[j]] = 0;
                else break;
            }
            sum = probs.Sum();
            for (int i = 0; i < n; i++) probs[i] /= sum;
        }

        double r = _rng.NextDouble();
        double acc = 0;
        int last = 0;
        for (int i = 0; i < n; i++)
        {
            if (probs[i] <= 0) continue;
            acc += probs[i];
            last = i;
            if (r < acc) return i;
        }
        return last;
    }

    public static int ArgMax(ReadOnlySpan<float> x)
    {
        int best = 0;
        for (int i = 1; i < x.Length; i++) if (x[i] > x[best]) best = i;
        return best;
    }
}
