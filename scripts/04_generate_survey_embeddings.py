import os
import csv
import json
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import sys

# Base directory configuration
BASE_DIR = os.path.dirname(__file__)

from helpers.identity_string_utils import create_survey_identity_string

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(text):
    """Get OpenAI embedding for text"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def generate_survey_embeddings():
    """Generate embeddings for all survey rows and save to disk"""
    survey_file = os.path.join(BASE_DIR, "../data/processed/02_music_survey_with_genres.csv")
    output_file = os.path.join(BASE_DIR, "../data/processed/survey_embeddings.json")

    print("Loading survey data...")
    with open(survey_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Generating embeddings for {len(rows)} survey responses...")

    embeddings_data = []

    for row in tqdm(rows, desc="Generating embeddings"):
        # Convert dict row to pandas Series for the function
        row_series = pd.Series(row)
        survey_text = create_survey_identity_string(row_series)
        embedding = get_embedding(survey_text)

        embeddings_data.append({
            'participant_id': row['participant_id'],
            'embedding': embedding,
            'data': {
                'Q1_Relationship_with_music': row['Q1_Relationship_with_music'],
                'Q2_Discovering_music': row['Q2_Discovering_music'],
                'Q3_artist_that_pulled_you_in': row['Q3_artist_that_pulled_you_in'],
                'Q4_Music_format_changes': row['Q4_Music_format_changes'],
                'Q9_Music_preference_these_days': row['Q9_Music_preference_these_days'],
                'Q10_Songs_by_AI': row['Q10_Songs_by_AI'],
                'Q11_Use_of_dead_artists_voice_feelings': row['Q11_Use_of_dead_artists_voice_feelings'],
                'Age': row['Age'],
                'Gender': row['Gender'],
                'Province': row['Province']
            }
        })

    # Save to JSON file
    print(f"\nSaving embeddings to {output_file}...")
    os.makedirs('data', exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f)

    print(f"✓ Successfully saved {len(embeddings_data)} embeddings!")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    generate_survey_embeddings()
