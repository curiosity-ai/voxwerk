namespace Voxwerk.Voices.English.Samples;

/// <summary>Registers the English voice catalog and the reference samples of every English voice.</summary>
public static class EnglishVoiceSamples
{
    /// <summary>Registers <see cref="Voxwerk.Voices.English.EnglishVoices"/> and every English audio package (idempotent).</summary>
    public static void RegisterAll()
    {
        Voxwerk.Voices.English.American.Characters.EnglishAmericanCharactersVoices.Register();
        Voxwerk.Voices.English.American.Commercial.EnglishAmericanCommercialVoices.Register();
        Voxwerk.Voices.English.American.Conversational.Female.EnglishAmericanConversationalFemaleVoices.Register();
        Voxwerk.Voices.English.American.Conversational.Male.EnglishAmericanConversationalMaleVoices.Register();
        Voxwerk.Voices.English.American.Narration.Female.EnglishAmericanNarrationFemaleVoices.Register();
        Voxwerk.Voices.English.American.Narration.Male.EnglishAmericanNarrationMaleVoices.Register();
        Voxwerk.Voices.English.American.Regional.EnglishAmericanRegionalVoices.Register();
        Voxwerk.Voices.English.British.Characters.EnglishBritishCharactersVoices.Register();
        Voxwerk.Voices.English.British.Commercial.EnglishBritishCommercialVoices.Register();
        Voxwerk.Voices.English.British.Conversational.EnglishBritishConversationalVoices.Register();
        Voxwerk.Voices.English.British.Narration.EnglishBritishNarrationVoices.Register();
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
