import pandas as pd
import os
import json
import sys

# Base directory configuration
BASE_DIR = os.path.dirname(__file__)

# Add helpers to path
sys.path.insert(0, os.path.join(BASE_DIR, 'helpers'))
from helpers.generate_image_prompt import create_image_prompt_from_survey

def generate_prompts_for_respondents(n=5):
    """
    Generate image prompts for n random survey respondents

    Args:
        n: Number of respondents to generate prompts for
    """

    # Load the processed survey data with genres
    input_csv = os.path.join(BASE_DIR, "../data/processed/02_music_survey_with_genres.csv")
    output_dir = os.path.join(BASE_DIR, "../data/analysis")
    output_csv = os.path.join(output_dir, "image_prompts.csv")
    output_json = os.path.join(output_dir, "image_prompts.json")

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Sample n random respondents
    sample_df = df.sample(n=min(n, len(df)))

    print(f"\nGenerating image prompts for {len(sample_df)} respondents...")

    results = []

    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        print(f"Processing {idx}/{len(sample_df)}: Participant {row['participant_id']}...", end="\r")

        try:
            # Generate image prompt
            image_prompt = create_image_prompt_from_survey(row)

            result = {
                'participant_id': row['participant_id'],
                'image_prompt': image_prompt,
            }

            results.append(result)

        except Exception as e:
            print(f"\nError processing participant {row['participant_id']}: {e}")
            continue

    print(f"\n\nSuccessfully generated {len(results)} image prompts!")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved CSV to: {output_csv}")

    # Save to JSON (more detailed)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to: {output_json}")

    return results_df


if __name__ == "__main__":
    # Generate prompts for 10 random respondents
    N_RESPONDENTS = 5

    results = generate_prompts_for_respondents(n=N_RESPONDENTS)

    print(f"\n✓ Done! Generated {len(results)} image prompts.")
