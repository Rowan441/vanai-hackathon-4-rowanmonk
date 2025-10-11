import pandas as pd
import os
from openai import OpenAI
import json
import requests
import time
import re


client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def validate_extracted_entities(original_text, entities_json):
    """
    Validate that extracted entities are correct.

    Returns: (is_valid, errors)
    """
    errors = []

    if pd.isna(entities_json) or not entities_json:
        return True, []  # Empty is valid

    # 1. Valid JSON
    try:
        data = json.loads(entities_json)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    if not isinstance(data, dict):
        return False, ["Root must be a dictionary"]

    # Check required fields
    if 'annotated_text' not in data:
        return False, ["Missing 'annotated_text' field"]

    if 'entities' not in data:
        return False, ["Missing 'entities' field"]

    annotated_text = data['annotated_text']
    entities = data['entities']

    if not isinstance(entities, list):
        return False, ["'entities' must be a list"]

    # 2. Extract annotations using regex
    pattern = r'\|\|(\{[^}]+\})([^|]+)\|\|'
    matches = list(re.finditer(pattern, annotated_text))

    # 3. Check for overlapping annotations
    positions = []
    for match in matches:
        start, end = match.span()
        for prev_start, prev_end in positions:
            if not (end <= prev_start or start >= prev_end):
                errors.append(f"Overlapping annotation at positions {start}-{end} and {prev_start}-{prev_end}")
        positions.append((start, end))

    # 4. Validate each annotation's JSON and fields
    for i, match in enumerate(matches):
        try:
            metadata = json.loads(match.group(1))
        except json.JSONDecodeError:
            errors.append(f"Invalid JSON in annotation {i}: {match.group(1)}")
            continue

        # Check required fields in metadata
        if 'type' not in metadata:
            errors.append(f"Annotation {i} missing 'type' field")
        elif metadata['type'] not in ['song', 'artist', 'album']:
            errors.append(f"Annotation {i} has invalid type: {metadata['type']}")

        if 'name' not in metadata:
            errors.append(f"Annotation {i} missing 'name' field")

    # 5. TEXT CONTENT MUST BE IDENTICAL (Most important!)
    # Remove all annotations to get the plain text
    plain_text = re.sub(pattern, r'\2', annotated_text)

    if plain_text.strip() != original_text.strip():
        errors.append(f"TEXT MISMATCH: Original text differs from annotated text content")

    # 6. Validate entity list matches annotations
    if len(entities) != len(matches):
        errors.append(f"Entity count mismatch: {len(entities)} entities but {len(matches)} annotations")

    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            errors.append(f"Entity {i} is not a dictionary")
            continue

        # Check fields
        required_fields = ['type', 'name', 'matched_text']
        for field in required_fields:
            if field not in entity:
                errors.append(f"Entity {i} missing '{field}' field")

    return len(errors) == 0, errors

# Spotify API setup
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

def get_spotify_token():
    """Get Spotify access token"""
    try:
        auth_response = requests.post('https://accounts.spotify.com/api/token', {
            'grant_type': 'client_credentials',
            'client_id': SPOTIFY_CLIENT_ID,
            'client_secret': SPOTIFY_CLIENT_SECRET,
        })
        auth_response.raise_for_status()
        auth_data = auth_response.json()
        return auth_data.get('access_token')
    except Exception as e:
        print(f"Error getting Spotify token: {e}")
        return None

def search_spotify(query, search_type='track'):
    """Search Spotify for a track, artist, or album"""
    token = get_spotify_token()
    if not token:
        return None

    headers = {'Authorization': f'Bearer {token}'}

    params = {
        'q': query,
        'type': search_type,
        'limit': 1
    }

    try:
        response = requests.get('https://api.spotify.com/v1/search', headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            if search_type == 'track' and data.get('tracks', {}).get('items'):
                item = data['tracks']['items'][0]
                return {
                    'name': item['name'],
                    'artist': item['artists'][0]['name'],
                    'url': item['external_urls']['spotify'],
                    'type': 'track'
                }
            elif search_type == 'artist' and data.get('artists', {}).get('items'):
                item = data['artists']['items'][0]
                return {
                    'name': item['name'],
                    'url': item['external_urls']['spotify'],
                    'type': 'artist'
                }
    except Exception as e:
        print(f"Error searching Spotify: {e}")

    return None

def extract_music_entities(text, question_context):
    """Use OpenAI to extract music entities by inline annotation"""
    if pd.isna(text) or not text.strip():
        return None

    prompt = f"""Add inline music entity annotations to this survey response.

Question: {question_context}
Response: {text}

Identify songs, artists, and albums in the text. For each entity, wrap it with special markers:
- Format: ||{{"type": one of ["song", "artist", "album"], "name": "correct_name", "artist": "correct_artist"}}original_text||

Rules:
- Only annotate entities you're confident about
- For "name" and "artist" fields: use the CORRECT, properly spelled version (for Spotify search)
- other than annotation and their json data keep text EXACTLY as written in the original (preserve typos, spacing, capitalization)
- Fix common issues in name/artist tag: apostrophes (L'il → Lil), spacing, capitalization
- For songs, include the artist if known
- Don't annotate the same text twice
- if there are no music entities within text return text completely unmodified and unhighlighted

Return ONLY the annotated text. Do not include explanations or commentary.

Example 1:
Input: 
my favourite song is probably Sleepwalking by Bring Me The Horizon
Output: 
my favourite song is probably ||{{"type": "song", "name": "Sleepwalking", "artist": "Bring Me The Horizon"}}Sleepwalking|| by ||{{"type": "artist", "name": "Bring Me The Horizon"}}Bring Me The Horizon||

Example 2:
Input: 
I love L'il nas x
Output: 
I love ||{{"type": "artist", "name": "Lil Nas X"}}L'il nas||

Example 3:
Input: 
Cant thing of any
Output: 
Cant thing of any

Now annotate this text: {text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a music entity annotation expert. Return the annotated text."},
            {"role": "user", "content": prompt}
        ],
        # temperature=0.3,
        max_tokens=800
    )

    annotated_text = response.choices[0].message.content.strip()

    # Parse the inline annotations
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

def process_survey_data(questions, test_rows=None, ):
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
                    # print(f"\nRow {idx} already completed, skipping...")
                    continue

                text = row[q['column']]

                if pd.isna(text) or not str(text).strip():
                    df.at[idx, q['new_column']] = None
                    continue

                if pd.notna(text) and text.strip():
                    print(f"\nProcessing row {idx}")
                    try:
                        # Extract entities with retry on validation failure
                        max_retries = 10
                        entities = None

                        for attempt in range(max_retries):
                            entities = extract_music_entities(text, q['question'])

                            if entities:
                                entities_json = json.dumps(entities)
                                is_valid, errors = validate_extracted_entities(text, entities_json)

                                if is_valid:
                                    # Add Spotify links
                                    entities = add_spotify_links(entities)
                                    break
                                else:
                                    print(f"Row {idx} attempt {attempt+1} validation failed: {errors}")
                                    if attempt < max_retries - 1:
                                        time.sleep(0.1)  # Brief pause before retry
                                    else:
                                        print(f"Row {idx} failed validation after {max_retries} attempts, skipping")
                                        # entities = None
                            else:
                                break

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

                time.sleep(0.1)  # Rate limiting for OpenAI

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
        },
    ]
if __name__ == "__main__":
    df = process_survey_data(questions)

    # Print sample results
    print("\n=== SAMPLE EXTRACTIONS ===")
    for col in ['Q3_extracted_entities', 'Q18_extracted_entities', 'Q16_extracted_entities', 'Q19_extracted_entities']:
        if col in df.columns:
            sample = df[df[col].notna()][col].head(3)
            print(f"\n{col}:")
            for item in sample:
                if item:
                    print(json.dumps(json.loads(item), indent=2))
