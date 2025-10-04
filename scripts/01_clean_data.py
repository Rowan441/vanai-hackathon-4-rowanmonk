import pandas as pd
import numpy as np
import os
import json
import time
from openai import OpenAI
from tqdm import tqdm

# Base directory configuration
BASE_DIR = os.path.dirname(__file__)

def load_data():
    """Load the music survey dataset"""
    data_path = os.path.join(BASE_DIR, "../data/raw/music_survey_data.csv")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded successfully")
        return df
    except FileNotFoundError:
        print(f"❌ Dataset not found at {data_path}")
        print("Please ensure the dataset is in the correct location")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
df = load_data()
df.info()
# Video data not available in dataset
df = df.drop(columns='Q16_Music_guilty_pleasure_Video_or_text') 
df = df.drop(columns='Q17_Video_share_concent')
# No non-null values (Inuit?) 
df = df.drop(columns='FirstNation_23_3')

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Fields to check for effort
EFFORT_FIELDS = {
    'Q3_artist_that_pulled_you_in': 'What was the first song or artist that really pulled you in?',
    'Q16_Music_guilty_pleasure_text_OE': "What’s your music guilty pleasure? Spill it 👀",
    'Q18_Life_theme_song': 'If your life had a theme song right now, what would it be — and why?',
    'Q19_Lyric_that_stuck_with_you': "What’s one song lyric that stuck with you or changed the way you see the world? Share it and who wrote it!"
}

def analyze_respondent(row):
    """
    Analyze all open-ended responses for a single respondent in ONE API call
    Returns: dict with overall assessment
    """
    # Prepare all responses
    responses_text = []
    for field, question in EFFORT_FIELDS.items():
        text = row.get(field, '')
        if pd.isna(text):
            text = '[NO ANSWER]'
        else:
            text = str(text).strip()
            if text == '':
                text = '[NO ANSWER]'
        responses_text.append(f"{field}: {question}\nResponse: \"{text}\"")

    prompt = f"""Analyze these 4 survey responses from one person and determine which are low-effort.

{chr(10).join(responses_text)}

A LOW-EFFORT response is:
- Single word or very short (e.g., "idk", "none", "nothing", "N/A", "pass")
- Nonsensical or random characters (e.g., "asdfgh", "...")
- Clearly not answering the question
- Joke/meme responses that show no genuine engagement
- [NO ANSWER] for required fields (Q16_Music_guilty_pleasure_text_OE can be null - that's acceptable)

Respond with ONLY a JSON object with this structure:
{{
  "Q3_artist_that_pulled_you_in": {{"is_low_effort": true/false}},
  "Q16_Music_guilty_pleasure_text_OE": {{"is_low_effort": true/false}},
  "Q18_Life_theme_song": {{"is_low_effort": true/false}},
  "Q19_Lyric_that_stuck_with_you": {{"is_low_effort": true/false}}
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a survey quality analyst. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )

        result_text = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
            result_text = result_text.strip()

        results = json.loads(result_text)

        # Process results
        low_effort_count = 0
        low_effort_fields = []

        for field in EFFORT_FIELDS.keys():
            field_result = results.get(field, {'is_low_effort': False})
            if field_result['is_low_effort']:
                low_effort_count += 1
                text = row.get(field, '')
                low_effort_fields.append({
                    'field': field,
                    'text': text,
                })

        # Consider low-effort if 2+ non-Q16 fields are low-effort
        non_q16_low_effort = sum(1 for f in low_effort_fields if f['field'] != 'Q16_Music_guilty_pleasure_text_OE')

        return {
            'is_low_effort': non_q16_low_effort >= 2,
            'low_effort_count': low_effort_count,
            'low_effort_fields': low_effort_fields
        }

    except Exception as e:
        print(f"\nError analyzing respondent: {e}")
        # On error, assume acceptable to be safe
        return {
            'is_low_effort': False,
            'low_effort_count': 0,
            'low_effort_fields': []
        }


print("\n" + "="*60)
print("FILTERING LOW-EFFORT RESPONSES")
print("="*60)

# Analyze each response
print("Analyzing response quality using OpenAI...")
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
    analysis = analyze_respondent(row)
    analysis['participant_id'] = row.get('participant_id', f'row_{idx}')
    results.append(analysis)
    df.at[idx, 'is_low_effort'] = analysis['is_low_effort']

# Filter out low-effort responses
original_count = len(df)
df = df[df['is_low_effort'] == False].copy()
df = df.drop(columns='is_low_effort')

print(f"\n✅ Quality filtering complete")
print(f"   Original responses: {original_count}")
print(f"   Low-effort removed: {original_count - len(df)} ({(original_count - len(df))/original_count*100:.1f}%)")
print(f"   High-quality kept: {len(df)} ({len(df)/original_count*100:.1f}%)")

# Save low-effort report
report_file = os.path.join(BASE_DIR, "../data/processed/low_effort_report.txt")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("LOW-EFFORT RESPONSE FILTERING REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total responses analyzed: {original_count}\n")
    f.write(f"Low-effort responses removed: {original_count - len(df)}\n")
    f.write(f"High-quality responses kept: {len(df)}\n\n")

    f.write("ALL LOW-EFFORT RESPONSES\n")
    f.write("-"*80 + "\n\n")

    low_effort_results = [r for r in results if r['is_low_effort']]
    for idx, result in enumerate(low_effort_results):
        f.write(f"{idx + 1}. Participant ID: {result['participant_id']}\n")
        f.write(f"   Low-effort fields: {result['low_effort_count']}/{len(EFFORT_FIELDS)}\n")
        for field_info in result['low_effort_fields']:
            f.write(f"   - {field_info['field']}: \"{field_info['text']}\"\n")
        f.write("\n")

print(f"   Report saved to: {report_file}")

# Save cleaned high-effort data
output_file = os.path.join(BASE_DIR, "../data/processed/01_music_survey_high_effort.csv")
df.to_csv(output_file, index=False)
print(f"\n✅ High-effort data saved to: {output_file}")
print(f"   Ready for next processing step!")