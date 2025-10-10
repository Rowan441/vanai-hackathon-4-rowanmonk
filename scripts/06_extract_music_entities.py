import pandas as pd
import os
from openai import OpenAI
import json
import requests
import time


client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Spotify API setup
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

def get_spotify_token():
    """Get Spotify access token"""
    auth_response = requests.post('https://accounts.spotify.com/api/token', {
        'grant_type': 'client_credentials',
        'client_id': SPOTIFY_CLIENT_ID,
        'client_secret': SPOTIFY_CLIENT_SECRET,
    })
    auth_data = auth_response.json()
    return auth_data['access_token']

def search_spotify(query, search_type='track'):
    """Search Spotify for a track, artist, or album"""
    token = get_spotify_token()
    headers = {'Authorization': f'Bearer {token}'}

    params = {
        'q': query,
        'type': search_type,
        'limit': 1
    }

    response = requests.get('https://api.spotify.com/v1/search', headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        if search_type == 'track' and data['tracks']['items']:
            item = data['tracks']['items'][0]
            return {
                'name': item['name'],
                'artist': item['artists'][0]['name'],
                'url': item['external_urls']['spotify'],
                'type': 'track'
            }
        elif search_type == 'artist' and data['artists']['items']:
            item = data['artists']['items'][0]
            return {
                'name': item['name'],
                'url': item['external_urls']['spotify'],
                'type': 'artist'
            }
    return None

def extract_music_entities(text, question_context):
    """Use OpenAI to extract music entities by inline annotation"""
    if pd.isna(text) or not text.strip():
        return None

    prompt = f"""Add inline music entity annotations to this survey response.

Question: {question_context}
Response: {text}

Identify songs, artists, and albums in the text. For each entity, wrap it with special markers:
- Format: ||{{"type": "song/artist/album", "name": "correct_name", "artist": "correct_artist"}}original_text||

Rules:
- Only annotate entities you're confident about
- For "name" and "artist" fields: use the CORRECT, properly spelled version (for Spotify search)
- For the text after }}: keep EXACTLY as written in the original (preserve typos, spacing, capitalization)
- Fix common issues in name/artist: apostrophes (L'il → Lil), spacing, capitalization
- For songs, include the artist if known
- Don't annotate the same text twice

Examples:
Input: "my favourite song is probably Sleepwalking by Bring Me The Horizon"
Output: "my favourite song is probably ||{{"type": "song", "name": "Sleepwalking", "artist": "Bring Me The Horizon"}}Sleepwalking|| by ||{{"type": "artist", "name": "Bring Me The Horizon"}}Bring Me The Horizon||"

Input: "I love L'il nas x"
Output: "I love ||{{"type": "artist", "name": "Lil Nas X"}}L'il nas||"

Now annotate this text: {text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a music entity annotation expert. Return the annotated text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )

    annotated_text = response.choices[0].message.content.strip()

    # Parse the inline annotations
    import re
    pattern = r'\|\|(\{[^}]+\})([^|]+)\|\|'

    entities = []
    for match in re.finditer(pattern, annotated_text):
        try:
            metadata = json.loads(match.group(1))
            matched_text = match.group(2)
            metadata['matched_text'] = matched_text
            entities.append(metadata)
        except:
            continue

    return {
        'annotated_text': annotated_text,
        'entities': entities
    }

def add_spotify_links(entities):
    """Add Spotify links to extracted entities"""
    if not entities or 'entities' not in entities:
        return entities

    for entity in entities['entities']:
        if entity['type'] == 'song':
            query = f"{entity['name']}"
            if 'artist' in entity:
                query += f" {entity['artist']}"
            spotify_result = search_spotify(query, 'track')
            if spotify_result:
                entity['spotify_url'] = spotify_result['url']
                entity['spotify_match'] = spotify_result['name']

        elif entity['type'] == 'artist':
            spotify_result = search_spotify(entity['name'], 'artist')
            if spotify_result:
                entity['spotify_url'] = spotify_result['url']
                entity['spotify_match'] = spotify_result['name']

        time.sleep(0.1)  # Rate limiting

    return entities

def process_survey_data(test_rows=None):
    """
    Process the survey CSV and extract music entities

    Args:
        test_rows: Optional int to limit processing to first N rows for testing
    """

    # Read the survey data
    df = pd.read_csv('data/processed/02_music_survey_with_genres.csv')

    # Limit to test rows if specified
    if test_rows:
        print(f"TEST MODE: Processing only first {test_rows} rows")
        df = df.head(test_rows)

    questions = [
        {
            'column': 'Q3_artist_that_pulled_you_in',
            'question': "What was the first song or artist that really pulled you in?",
            'new_column': 'Q3_extracted_entities'
        },
        {
            'column': 'Q18_Life_theme_song',
            'question': "If your life had a theme song right now, what would it be — and why?",
            'new_column': 'Q18_extracted_entities'
        },
        {
            'column': 'Q16_Music_guilty_pleasure_text_OE',
            'question': "What's your music guilty pleasure? Spill it 👀",
            'new_column': 'Q16_extracted_entities'
        },
        {
            'column': 'Q19_Lyric_that_stuck_with_you',
            'question': "What's one song lyric that stuck with you or changed the way you see the world? Share it and who wrote it!",
            'new_column': 'Q19_extracted_entities'
        }
    ]

    output_path = 'data/processed/03_music_survey_with_extracted_entities.csv'

    # Check if we have partial progress to resume from
    if os.path.exists(output_path):
        print(f"Found existing output file. Loading progress...")
        existing_df = pd.read_csv(output_path)
        for q in questions:
            if q['new_column'] in existing_df.columns:
                df[q['new_column']] = existing_df[q['new_column']]
                print(f"  Loaded existing data for {q['column']}")

    try:
        for q in questions:
            # Skip if already completed
            if q['new_column'] in df.columns and df[q['new_column']].notna().all():
                print(f"\n{q['column']} already completed, skipping...")
                continue

            print(f"\nProcessing: {q['column']}")

            # Initialize column if doesn't exist
            if q['new_column'] not in df.columns:
                df[q['new_column']] = None

            for idx, row in df.iterrows():
                # Skip if already processed
                if pd.notna(df.at[idx, q['new_column']]):
                    continue

                text = row[q['column']]

                if pd.notna(text) and text.strip():
                    try:
                        # Extract entities
                        entities = extract_music_entities(text, q['question'])

                        # Add Spotify links
                        if entities:
                            entities = add_spotify_links(entities)

                        df.at[idx, q['new_column']] = json.dumps(entities) if entities else None
                    except Exception as e:
                        print(f"Error processing row {idx}: {e}")
                        df.at[idx, q['new_column']] = None
                else:
                    df.at[idx, q['new_column']] = None

                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(df)} rows")
                    # Save checkpoint every 10 rows
                    df.to_csv(output_path, index=False)

                time.sleep(0.2)  # Rate limiting for OpenAI

            # Save after each question
            df.to_csv(output_path, index=False)
            print(f"Saved checkpoint for {q['column']}")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving progress...")
        df.to_csv(output_path, index=False)
        print(f"Progress saved to: {output_path}")
        print("Run script again to resume from where you left off.")
        raise
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        print("Saving progress...")
        df.to_csv(output_path, index=False)
        print(f"Progress saved to: {output_path}")
        raise

    # Final save
    df.to_csv(output_path, index=False)
    print(f"\nCompleted! Results saved to: {output_path}")

    return df

if __name__ == "__main__":
    df = process_survey_data()

    # Print sample results
    print("\n=== SAMPLE EXTRACTIONS ===")
    for col in ['Q3_extracted_entities', 'Q18_extracted_entities', 'Q16_extracted_entities', 'Q19_extracted_entities']:
        if col in df.columns:
            sample = df[df[col].notna()][col].head(3)
            print(f"\n{col}:")
            for item in sample:
                if item:
                    print(json.dumps(json.loads(item), indent=2))
