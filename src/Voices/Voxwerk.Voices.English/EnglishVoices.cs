namespace Voxwerk.Voices.English;

/// <summary>Registers every English voice package.</summary>
public static class EnglishVoices
{
    /// <summary>Registers all English regional voice packs with <see cref="VoiceRegistry"/> (idempotent).</summary>
    public static void RegisterAll()
    {
        Voxwerk.Voices.English.American.EnglishAmericanVoices.Register();
        Voxwerk.Voices.English.British.EnglishBritishVoices.Register();
        Voxwerk.Voices.English.Irish.EnglishIrishVoices.Register();
        Voxwerk.Voices.English.Scottish.EnglishScottishVoices.Register();
        Voxwerk.Voices.English.Canadian.EnglishCanadianVoices.Register();
        Voxwerk.Voices.English.Australian.EnglishAustralianVoices.Register();
        Voxwerk.Voices.English.NewZealand.EnglishNewZealandVoices.Register();
        Voxwerk.Voices.English.Indian.EnglishIndianVoices.Register();
        Voxwerk.Voices.English.African.EnglishAfricanVoices.Register();
        Voxwerk.Voices.English.Asian.EnglishAsianVoices.Register();
        Voxwerk.Voices.English.European.EnglishEuropeanVoices.Register();
        Voxwerk.Voices.English.International.EnglishInternationalVoices.Register();
    }
}
