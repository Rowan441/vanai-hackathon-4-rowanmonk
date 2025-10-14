import pandas as pd
import os
import requests
import time

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

def search_spotify_artist(artist_name):
    """Search Spotify for an artist and return their URL"""
    if pd.isna(artist_name) or not str(artist_name).strip():
        return None

    token = get_spotify_token()
    if not token:
        return None

    headers = {'Authorization': f'Bearer {token}'}

    params = {
        'q': artist_name,
        'type': 'artist',
        'limit': 1
    }

    try:
        response = requests.get('https://api.spotify.com/v1/search', headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            if data.get('artists', {}).get('items'):
                item = data['artists']['items'][0]
                return item['external_urls']['spotify']
    except Exception as e:
        print(f"Error searching Spotify for '{artist_name}': {e}")

    return None

def process_favourite_artists():
    """
    Load survey data and add Spotify URLs for favourite artists
    """

    # Read the survey data
    input_path = 'data/processed/03_music_survey_with_extracted_entities.csv'
    output_path = 'data/processed/04_music_survey_with_artist_urls.csv'

    df = pd.read_csv(input_path)

    # Check if we have partial progress to resume from
    if os.path.exists(output_path):
        print(f"Found existing output file. Loading progress...")
        existing_df = pd.read_csv(output_path)
        if 'extracted_favourite_band_spotify_url' in existing_df.columns:
            df['extracted_favourite_band_spotify_url'] = existing_df['extracted_favourite_band_spotify_url']
            print(f"  Loaded existing Spotify URL data")

    # Initialize column if doesn't exist
    if 'extracted_favourite_band_spotify_url' not in df.columns:
        df['extracted_favourite_band_spotify_url'] = None

    try:
        print(f"\nProcessing favourite artists...")

        for idx, row in df.iterrows():
            # Skip if already processed
            if pd.notna(df.at[idx, 'extracted_favourite_band_spotify_url']):
                continue

            artist_name = row.get('extracted_favourite_band')

            if pd.isna(artist_name) or not str(artist_name).strip():
                df.at[idx, 'extracted_favourite_band_spotify_url'] = None
                continue

            print(f"\nRow {idx}: Searching for '{artist_name}'")

            try:
                spotify_url = search_spotify_artist(artist_name)
                df.at[idx, 'extracted_favourite_band_spotify_url'] = spotify_url

                if spotify_url:
                    print(f"  ✓ Found: {spotify_url}")
                else:
                    print(f"  ✗ Not found")

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                df.at[idx, 'extracted_favourite_band_spotify_url'] = None

            # Save checkpoint every 10 rows
            if (idx + 1) % 10 == 0:
                print(f"\nProcessed {idx + 1}/{len(df)} rows")
                df.to_csv(output_path, index=False)

            time.sleep(0.1)  # Rate limiting for Spotify API

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

    # Print statistics
    total_artists = df['extracted_favourite_band'].notna().sum()
    found_urls = df['extracted_favourite_band_spotify_url'].notna().sum()
    print(f"\n=== STATISTICS ===")
    print(f"Total favourite artists: {total_artists}")
    print(f"Spotify URLs found: {found_urls}")
    print(f"Success rate: {found_urls/total_artists*100:.1f}%")

    return df

if __name__ == "__main__":
    df = process_favourite_artists()

    # Print sample results
    print("\n=== SAMPLE RESULTS ===")
    sample = df[df['extracted_favourite_band_spotify_url'].notna()][
        ['extracted_favourite_band', 'extracted_favourite_band_spotify_url']
    ].head(5)
    print(sample.to_string(index=False))
