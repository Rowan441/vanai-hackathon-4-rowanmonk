import pandas as pd
import openai
import os
from typing import Optional
from enum import Enum
import json
import time

# Base directory configuration
BASE_DIR = os.path.dirname(__file__)

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MusicGenre(Enum):
    """Predefined music genres for classification."""
    ROCK = "rock"
    POP = "pop"
    HIP_HOP = "hip hop"
    RAP = "rap"
    RNB = "r&b"
    COUNTRY = "country"
    INDIE = "indie"
    ALTERNATIVE = "alternative"
    METAL = "metal"
    PUNK = "punk"
    ELECTRONIC = "electronic"
    EDM = "edm"
    JAZZ = "jazz"
    BLUES = "blues"
    FOLK = "folk"
    CLASSICAL = "classical"
    REGGAE = "reggae"
    LATIN = "latin"
    KPOP = "k-pop"
    OTHER = "other"

def extract_genre_and_band(row: pd.Series) -> dict:
    """
    Extract genre and favourite band from survey responses using OpenAI API.

    Fields used:
    - Q3_artist_that_pulled_you_in (open ended)
    - Q9_Music_preference_these_days (single choice)
    - Q16_Music_guilty_pleasure_text_OE (open ended, many null)
    - Q18_Life_theme_song (open ended)
    - Q19_Lyric_that_stuck_with_you (open ended)
    """

    # Prepare the context from survey responses
    context_parts = []

    if pd.notna(row.get('Q3_artist_that_pulled_you_in')):
        context_parts.append(f"Artist that pulled them in: {row['Q3_artist_that_pulled_you_in']}")

    if pd.notna(row.get('Q9_Music_preference_these_days')):
        context_parts.append(f"Current music preference: {row['Q9_Music_preference_these_days']}")

    if pd.notna(row.get('Q16_Music_guilty_pleasure_text_OE')):
        context_parts.append(f"Guilty pleasure: {row['Q16_Music_guilty_pleasure_text_OE']}")

    if pd.notna(row.get('Q18_Life_theme_song')):
        context_parts.append(f"Life theme song: {row['Q18_Life_theme_song']}")

    if pd.notna(row.get('Q19_Lyric_that_stuck_with_you')):
        context_parts.append(f"Memorable lyric: {row['Q19_Lyric_that_stuck_with_you']}")

    if not context_parts:
        return {"genre": None, "favourite_band": None, "confidence": "low"}

    context = "\n".join(context_parts)

    # Get allowed genre values
    allowed_genres = [genre.value for genre in MusicGenre]

    # Create the prompt
    prompt = f"""Based on the following music survey responses, extract:
1. The person's likely favourite music genre (must be one of: {', '.join(allowed_genres)})
2. Their favourite band or artist

Survey responses:
{context}

Respond in JSON format with these fields:
- "genre": string (must be exactly one of: {', '.join(allowed_genres)}, or null if cannot determine)
- "favourite_band": string (the identified favourite band/artist, or null if cannot determine)
- "confidence": string ("high", "medium", or "low")

Keep your response concise and only return the JSON object."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a music analyst who extracts genre and artist preferences from survey data. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]

        result = json.loads(result_text.strip())

        # Validate genre against enum
        if result.get('genre') is not None:
            genre_lower = result['genre'].lower()
            valid_genres = [g.value for g in MusicGenre]
            if genre_lower not in valid_genres:
                # Try to find closest match or set to "other"
                result['genre'] = "other"
                result['confidence'] = "low"

        return result

    except Exception as e:
        print(f"Error processing row: {e}")
        return {"genre": None, "favourite_band": None, "confidence": "error", "error": str(e)}


def process_survey_data(csv_path: str, output_path: str, sample_size: Optional[int] = None):
    """
    Process the music survey data and extract genre/band information.

    Args:
        csv_path: Path to the input CSV file
        output_path: Path to save the output CSV file
        sample_size: If provided, only process this many rows (useful for testing)
    """

    # Load the data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, ignore_index=True)

    if sample_size:
        df = df.head(sample_size)
        print(f"Processing sample of {sample_size} rows")
    else:
        print(f"Processing {len(df)} rows")

    # Process each row
    results = []
    for idx, row in df.iterrows():
        print(f"Processing row {idx + 1}/{len(df)}...", end="\r")

        result = extract_genre_and_band(row)
        results.append(result)

        # Add small delay to avoid rate limits
        time.sleep(0.5)

    print("\nProcessing complete!")

    # Add results to dataframe
    df['extracted_genre'] = [r.get('genre') for r in results]
    df['extracted_favourite_band'] = [r.get('favourite_band') for r in results]
    df['extraction_confidence'] = [r.get('confidence') for r in results]

    # Save results
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Total rows processed: {len(df)}")
    print(f"Genres extracted: {df['extracted_genre'].notna().sum()}")
    print(f"Bands extracted: {df['extracted_favourite_band'].notna().sum()}")
    print(f"\nConfidence distribution:")
    print(df['extraction_confidence'].value_counts())

    return df


if __name__ == "__main__":
    # File paths
    input_csv = os.path.join(BASE_DIR, "../data/processed/01_music_survey_high_effort.csv")
    output_csv = os.path.join(BASE_DIR, "../data/processed/02_music_survey_with_genres.csv")

    # Test with a small sample first (remove or set to None to process all)
    SAMPLE_SIZE = None

    # Process the data
    df_result = process_survey_data(input_csv, output_csv, sample_size=SAMPLE_SIZE)

    # Display some examples
    print("\n--- Sample Results ---")
    print(df_result[['participant_id',
        'Q3_artist_that_pulled_you_in',
        'Q9_Music_preference_these_days',
        'Q16_Music_guilty_pleasure_text_OE',
        'Q18_Life_theme_song',
        'Q19_Lyric_that_stuck_with_you',
        'extracted_genre',
        'extracted_favourite_band',
        'extraction_confidence']].head(10))