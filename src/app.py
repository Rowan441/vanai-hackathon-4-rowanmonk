from flask import Flask, render_template, jsonify, request
import json
import csv
import os
import sys
import numpy as np
from collections import Counter
from openai import OpenAI
import re


# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from helpers.generate_image_prompt import AISpectrumLevel, IntensityLevel, SocialityLevel, generate_avatar_prompt
from helpers.identity_string_utils import create_user_identity_string
from helpers.generate_image_prompt import create_image_prompt_from_survey
from models import RespondentProfile, MatchResult, QuestionnaireResponse

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Base directory configuration
BASE_DIR = os.path.dirname(__file__)

app = Flask(__name__)

def load_survey_data():
    """Load survey data from CSV with extracted entities"""
    # Load data with extracted entities
    entities_file = os.path.join(BASE_DIR, "../data/processed/03_music_survey_with_extracted_entities.csv")
    with open(entities_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def convert_entities_to_html(text, entities_json):
    """Convert text with entity annotations to HTML with Spotify links"""
    if not entities_json or entities_json == 'nan' or str(entities_json).strip() == '':
        return text

    try:
        data = json.loads(entities_json)
    except (json.JSONDecodeError, TypeError):
        return text

    if not data or 'annotated_text' not in data:
        return text

    # Start with the annotated text
    html = data['annotated_text']

    # Replace each entity marker with HTML
    pattern = r'\|\|(\{[^}]+\})([^|]+)\|\|'

    def replace_entity(match):
        try:
            metadata = json.loads(match.group(1))
            matched_text = match.group(2)

            # Find corresponding entity with spotify_url
            spotify_url = None
            for entity in data.get('entities', []):
                if entity.get('matched_text') == matched_text:
                    spotify_url = entity.get('spotify_url')
                    break

            if spotify_url:
                return f'<a href="{spotify_url}" target="_blank" class="music-entity">{matched_text}</a>'
            else:
                return matched_text
        except:
            return match.group(2)

    html = re.sub(pattern, replace_entity, html)
    return html

def load_embeddings():
    """Load embeddings data"""
    embeddings_file = os.path.join(BASE_DIR, "../data/processed/survey_embeddings.json")
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

@app.route('/')
def index():
    """Main Page"""
    return render_template('index.html')

@app.route('/stats')
def stats():
    """Dashboard page"""
    return render_template('stats.html')

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.route("/submit_answers", methods=["POST"])
def submit_answers():
    try:
        data = request.get_json()

        # Create identity string from user answers
        identity_string = create_user_identity_string(data)
        print("User identity string:", identity_string)

        # Generate user embedding
        response = client.embeddings.create(
            input=identity_string,
            model="text-embedding-3-small"
        )
        user_embedding = response.data[0].embedding

        # Load pre-computed survey embeddings
        embeddings_data = load_embeddings()

        if not embeddings_data:
            return jsonify({"status": "error", "message": "No embeddings found"}), 500

        # Find best match with cosine similarity
        best_match = None
        best_similarity = -1

        for entry in embeddings_data:
            similarity = cosine_similarity(user_embedding, entry['embedding'])
            if similarity > best_similarity:
                # print(entry["participant_id"])
                # print(similarity)
                best_similarity = similarity
                best_match = entry['participant_id']

        print(f"Best match: {best_match} with similarity: {best_similarity}")

        # Load full survey data to get matched response details
        survey_data = load_survey_data()
        matched_response = next((row for row in survey_data if row['participant_id'] == best_match), None)

        if not matched_response:
            return jsonify({"status": "error", "message": "Match not found in survey data"}), 500

        # Build discovery methods string
        discovery_methods = []
        if matched_response.get('Q7_New_music_discover_1'): discovery_methods.append("TikTok/Reels")
        if matched_response.get('Q7_New_music_discover_2'): discovery_methods.append("Streaming playlists")
        if matched_response.get('Q7_New_music_discover_3'): discovery_methods.append("Friend recommendations")
        if matched_response.get('Q7_New_music_discover_4'): discovery_methods.append("Movie/TV soundtracks")
        if matched_response.get('Q7_New_music_discover_5'): discovery_methods.append("Shazam")
        if matched_response.get('Q7_New_music_discover_6'): discovery_methods.append("Music blogs / critics")
        if matched_response.get('Q7_New_music_discover_7'): discovery_methods.append("Just replays their favourites")

        # Build listening contexts
        listening_contexts = []
        if matched_response.get('Q8_Music_listen_time_GRID_1') in ['Often', 'Always']: listening_contexts.append("Waking up")
        if matched_response.get('Q8_Music_listen_time_GRID_2') in ['Often', 'Always']: listening_contexts.append("Commuting")
        if matched_response.get('Q8_Music_listen_time_GRID_3') in ['Often', 'Always']: listening_contexts.append("Working out")
        if matched_response.get('Q8_Music_listen_time_GRID_4') in ['Often', 'Always']: listening_contexts.append("Cooking")
        if matched_response.get('Q8_Music_listen_time_GRID_5') in ['Often', 'Always']: listening_contexts.append("Cleaning")
        if matched_response.get('Q8_Music_listen_time_GRID_6') in ['Often', 'Always']: listening_contexts.append("Unwinding")

        # Build music acheivements
        acheivements = []
        if matched_response.get('Q12_Music_bingo_1'): acheivements.append("Made a breakup playlists")
        if matched_response.get('Q12_Music_bingo_2'): acheivements.append("Played DJ on road trip")
        if matched_response.get('Q12_Music_bingo_3'): acheivements.append("Used music to hype themselves up")
        if matched_response.get('Q12_Music_bingo_4'): acheivements.append("Cried to a sad song")
        if matched_response.get('Q12_Music_bingo_5'): acheivements.append("Shared a song to flirt")
        if matched_response.get('Q12_Music_bingo_6'): acheivements.append("Made a playlist just for the vibes")
        if matched_response.get('Q12_Music_bingo_7'): acheivements.append("Replayed the same song 10+ times")

        # Build sharing methods
        sharing_methods = []
        if matched_response.get('Q13_Share_the_music_you_love_1'): sharing_methods.append("Texting Music Links")
        if matched_response.get('Q13_Share_the_music_you_love_2'): sharing_methods.append("Group chats")
        if matched_response.get('Q13_Share_the_music_you_love_3'): sharing_methods.append("Social media")
        if matched_response.get('Q13_Share_the_music_you_love_4'): sharing_methods.append("Shares playlists")
        if matched_response.get('Q13_Share_the_music_you_love_5'): sharing_methods.append("In-person")
        if matched_response.get('Q13_Share_the_music_you_love_6'): sharing_methods.append("Doesn't share music")

        # Convert extracted entities to HTML with Spotify links
        first_artist_html = convert_entities_to_html(
            matched_response.get('Q3_artist_that_pulled_you_in', 'N/A'),
            matched_response.get('Q3_extracted_entities')
        )
        guilty_pleasure_html = convert_entities_to_html(
            matched_response.get('Q16_Music_guilty_pleasure_text_OE', 'N/A'),
            matched_response.get('Q16_extracted_entities')
        )
        theme_song_html = convert_entities_to_html(
            matched_response.get('Q18_Life_theme_song', 'N/A'),
            matched_response.get('Q18_extracted_entities')
        )
        favorite_lyric_html = convert_entities_to_html(
            matched_response.get('Q19_Lyric_that_stuck_with_you', 'N/A'),
            matched_response.get('Q19_extracted_entities')
        )
        favorite_band_html = convert_entities_to_html(
            matched_response.get('extracted_favourite_band', 'N/A'),
            matched_response.get('extracted_favourite_band_entities')
        )

        profile = RespondentProfile(
            age=matched_response.get('Age', 'N/A'),
            gender=matched_response.get('Gender', 'N/A'),
            location=", ".join(filter(None, [matched_response.get('CMA'), matched_response.get('Province')])) or "N/A",
            relationship_with_music=matched_response.get('Q1_Relationship_with_music', 'N/A'),
            discovering_music=matched_response.get('Q2_Discovering_music', 'N/A'),
            first_song_artist_love=first_artist_html,
            # format_change=matched_response.get('Q4_Music_format_changes', 'N/A'),
            # format_change_memory=matched_response.get('Q5_Music_format_change_impact', 'N/A'),
            # format_change_feelings=matched_response.get('Q6_Music_format_change_feelings', 'N/A'),
            discovery_methods=', '.join(discovery_methods) if discovery_methods else 'N/A',
            listening_contexts=', '.join(listening_contexts) if listening_contexts else 'N/A',
            current_preference=matched_response.get('Q9_Music_preference_these_days', 'N/A'),
            ai_songs=matched_response.get('Q10_Songs_by_AI', 'N/A'),
            dead_artists_voice=matched_response.get('Q11_Use_of_dead_artists_voice_feelings', 'N/A'),
            music_achievements=', '.join(acheivements) if acheivements else 'N/A',
            sharing_methods=', '.join(sharing_methods) if sharing_methods else 'N/A',
            friend_shares_reaction=matched_response.get('Q14_Friend_shares_a_song', 'N/A'),
            guilty_pleasure_attitude=matched_response.get('Q15_Music_guilty_pleasure', 'N/A'),
            guilty_pleasure_song=guilty_pleasure_html,
            theme_song=theme_song_html,
            favorite_lyric=favorite_lyric_html,
            favorite_genre=matched_response.get('extracted_genre', 'N/A'),
            favorite_band=favorite_band_html
        )

        match = MatchResult(
            participant_id=best_match,
            similarity_score=float(best_similarity),
            profile=profile
        )

        response = QuestionnaireResponse(
            status="success",
            match=match
        )

        return jsonify(response.model_dump()), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/analyze_match", methods=["POST"])
def analyze_match():
    """Analyze which fields are most similar between user and match"""
    try:
        data = request.get_json()
        user_answers = data.get('user_answers', {})
        match_profile = data.get('match_profile', {})

        # Create JSON objects for comparison
        user_profile_json = {
            "What's your relationship with music like?": user_answers.get('q1', 'N/A'),
            "How did you first discover music you loved?": user_answers.get('q2', 'N/A'),
            "What kind of music are you into these days?": user_answers.get('q3', 'N/A'),
            "Real talk - how do you feel about AI making music": user_answers.get('q4', 'N/A'),
            "In what situations are you listening music the most?": user_answers.get('q5', 'N/A'),
            "What is your absolute favourite band / artist and what do you love about them??": user_answers.get('q6', 'N/A')
        }

        import json as json_lib
        user_json_str = json_lib.dumps(user_profile_json, indent=2)
        match_json_str = json_lib.dumps(match_profile, indent=2)

        # Create comparison prompt
        prompt = f"""Analyze the similarities between a user's music taste quiz answers and their matched survey respondent.

USER'S ANSWERS:
{user_json_str}

MATCHED PERSON'S FULL PROFILE:
{match_json_str}

Identify ONLY truly meaningful similarities. Return 0-3 insights maximum - quality over quantity. If there are no strong connections, return an empty insights array.

Requirements for each insight:
- Must reference specific USER'S ANSWERS as evidence
- Must be conversational and engaging
- Must not reference the MATCHED PERSON field names
- Must relate specifically to the CONTENT of the selected MATCHED PERSON field

Format your response as a JSON object:
{{
  "summary": "2-4 sentence summary explaining the match quality. If weak match, acknowledge it honestly. Write directly to the user using 'you'.",
  "insights": [
    {{"field": "MATCHED PERSON field name", "insight": "Explanation that quotes or references the USER'S ANSWERS as proof of the connection. "}}
  ]
}}

Example of a good insight: "You both see music as a form of self-expression - you said music is 'part of your identity' and they describe it as 'essential to who I am'."
Example of a bad insight: "You both like music." (too vague, no evidence)

Be honest, selective, and only highlight genuine connections."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a music taste analyst who finds meaningful connections for people. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )

        result_text = response.choices[0].message.content.strip()

        # Remove markdown code blocks if present
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
            result_text = result_text.strip()

        import json as json_lib
        analysis_result = json_lib.loads(result_text)

        return jsonify({
            "status": "success",
            "summary": analysis_result.get("summary", ""),
            "insights": analysis_result.get("insights", [])
        }), 200

    except Exception as e:
        print(f"Error analyzing match: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.route("/generate_avatar", methods=["POST"])
def generate_avatar():
    """Generate AI avatar for matched profile using DALL-E"""
    try:
        data = request.get_json()
        participant_id = data.get('participant_id')

        if not participant_id:
            return jsonify({
                "status": "error",
                "message": "participant_id required"
            }), 400

        # Load survey data to find the matched response
        survey_data = load_survey_data()
        matched_response = next(
            (r for r in survey_data if r['participant_id'] == participant_id),
            None
        )

        if not matched_response:
            return jsonify({
                "status": "error",
                "message": "Participant not found"
            }), 404

        # Generate image prompt from survey data
        image_prompt = create_image_prompt_from_survey(matched_response)

        print(f"Generated prompt: {image_prompt}")

        # Generate image using gpt with base64 response
        response = client.images.generate(
            model="gpt-image-1",
            prompt=image_prompt,
            size="1024x1024",
            quality="medium",
            n=1,
        )

        # Get base64 encoded image data
        image_b64 = response.data[0].b64_json

        # Create data URI for frontend
        image_data_uri = f"data:image/png;base64,{image_b64}"

        return jsonify({
            "status": "success",
            "image_url": image_data_uri,
            "prompt": image_prompt
        }), 200

    except Exception as e:
        print(f"Error generating avatar: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.route('/generate_user_avatar', methods=["POST"])
def generate_user_avatar():
    try:
        data = request.get_json()
        physical_description = data.get('physical_description')
        user_answers = data.get('user_answers', {})

        # Use GPT to extract structured attributes from user responses
        extraction_prompt = f"""Analyze these music questionnaire responses and extract the following attributes:

User Responses:
- Q1: What's your relationship with music like?: {user_answers.get('q1', 'N/A')}
- Q2: How did you first discover music you loved?: {user_answers.get('q2', 'N/A')}
- Q3: What kind of music are you into these days?: {user_answers.get('q3', 'N/A')}
- Q4: Real talk - how do you feel about AI making music?: {user_answers.get('q4', 'N/A')}
- Q5: In what situations are you listening to music the most?: {user_answers.get('q5', 'N/A')}
- Q6: What is your absolute favourite band / artist and what do you love about them?: {user_answers.get('q6', 'N/A')}

Extract and return ONLY a JSON object with these fields:
{{
    "ai_level": "embracer" | "curious" | "uncertain" | "rejector",
    "intensity_level": "obsessed" | "engaged" | "casual" | "minimal",
    "sociality_level": "active_curator" | "social_listener" | "casual_sharer" | "hoarder",
    "favourite_genre": "rock" | "pop" | "hip hop" | "indie" | "electronic" | etc. (or null if unclear),
    "favourite_band": "Band/Artist Name" (extract from q6, or null if not mentioned)
}}

Guidelines:
- ai_level: Based on q4 - how they feel about AI music
- intensity_level: Based on q1 and q5 - how much music is part of their life
- sociality_level: Based on q5 and overall tone - how much they share music with others
- favourite_genre: Extract from q3 or q6
- favourite_band: Extract exact band/artist name from q6"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a music preference analyzer. Return only valid JSON, no markdown formatting."},
                {"role": "user", "content": extraction_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        # Parse GPT response - clean markdown code blocks if present
        content = response.choices[0].message.content.strip()
        print(f"Raw GPT response: {content}")

        # Remove markdown code blocks
        if content.startswith('```'):
            # Remove opening ```json or ```
            content = content.split('\n', 1)[1] if '\n' in content else content[3:]
            # Remove closing ```
            if content.endswith('```'):
                content = content.rsplit('```', 1)[0]
            content = content.strip()

        print(f"Cleaned response: {content}")
        extracted = json.loads(content)

        # Map to enum values
        ai_level_map = {
            "embracer": AISpectrumLevel.EMBRACER,
            "curious": AISpectrumLevel.CURIOUS,
            "uncertain": AISpectrumLevel.UNCERTAIN,
            "rejector": AISpectrumLevel.REJECTOR
        }

        intensity_level_map = {
            "obsessed": IntensityLevel.OBSESSED,
            "engaged": IntensityLevel.ENGAGED,
            "casual": IntensityLevel.CASUAL,
            "minimal": IntensityLevel.MINIMAL
        }

        sociality_level_map = {
            "active_curator": SocialityLevel.ACTIVE_CURATOR,
            "social_listener": SocialityLevel.SOCIAL_LISTENER,
            "casual_sharer": SocialityLevel.CASUAL_SHARER,
            "hoarder": SocialityLevel.HOARDER
        }

        ai_level = ai_level_map.get(extracted.get('ai_level', 'uncertain'), AISpectrumLevel.UNCERTAIN)
        intensity_level = intensity_level_map.get(extracted.get('intensity_level', 'casual'), IntensityLevel.CASUAL)
        sociality_level = sociality_level_map.get(extracted.get('sociality_level', 'casual_sharer'), SocialityLevel.CASUAL_SHARER)
        favourite_genre = extracted.get('favourite_genre')
        favourite_band = extracted.get('favourite_band')

        # Generate Musical Avatar Image Prompt using user's physical description
        avatar_prompt = generate_avatar_prompt(
            physical_desc=physical_description,
            ai_level=ai_level,
            intensity_level=intensity_level,
            sociality_level=sociality_level,
            favourite_genre=favourite_genre,
            favourite_band=favourite_band
        )

        print(f"Generated user avatar prompt: {avatar_prompt}")

        try:
            image_response = client.images.generate(
                model="gpt-image-1",
                prompt=avatar_prompt,
                size="1024x1024",
                quality="medium",
                n=1,
            )

             # Get base64 encoded image data
            image_b64 = image_response.data[0].b64_json

            # Create data URI for frontend
            image_data_uri = f"data:image/png;base64,{image_b64}"

            return jsonify({
                'status': 'success',
                'image_url': image_data_uri,
                'prompt': avatar_prompt
            })

        except Exception as dalle_error:
            error_str = str(dalle_error)

            # Check if it's a safety/moderation error
            if 'safety system' in error_str.lower() or 'moderation_blocked' in error_str.lower() or 'content_policy' in error_str.lower():
                print(f"Safety block error: {dalle_error}")
                return jsonify({
                    'status': 'error',
                    'message': 'Avatar generation blocked: Your description may contain copyrighted characters or inappropriate content. Please try a different description.',
                    'error_type': 'safety_block'
                }), 400
            else:
                # Other DALL-E errors
                print(f"DALL-E error: {dalle_error}")
                raise dalle_error

    except Exception as e:
        print(f"Error generating user avatar: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Unable to generate avatar. Please try again or contact support if the issue persists.'
        }), 500

@app.route('/api/stats')
def get_stats():
    """Get summary statistics about the survey data"""
    data = load_survey_data()

    stats = {
        'total_responses': len(data),
        'demographics': {
            'age_groups': dict(Counter(row['AgeGroup_Broad'] for row in data if row.get('AgeGroup_Broad'))),
            'gender': dict(Counter(row['Gender'] for row in data if row.get('Gender'))),
            'provinces': dict(Counter(row['Province'] for row in data if row.get('Province')))
        },
        'music_preferences': {
            'relationship': dict(Counter(row['Q1_Relationship_with_music'] for row in data if row.get('Q1_Relationship_with_music'))),
            'discovery': dict(Counter(row['Q2_Discovering_music'] for row in data if row.get('Q2_Discovering_music'))),
            'current_preference': dict(Counter(row['Q9_Music_preference_these_days'] for row in data if row.get('Q9_Music_preference_these_days'))),
            'ai_songs': dict(Counter(row['Q10_Songs_by_AI'] for row in data if row.get('Q10_Songs_by_AI'))),
            'dead_artists_voice': dict(Counter(row['Q11_Use_of_dead_artists_voice_feelings'] for row in data if row.get('Q11_Use_of_dead_artists_voice_feelings')))
        },
        'format_changes': dict(Counter(row['Q4_Music_format_changes'] for row in data if row.get('Q4_Music_format_changes')))
    }

    return jsonify(stats)

@app.route('/api/responses')
def get_responses():
    """Get all survey responses with optional filtering"""
    data = load_survey_data()

    # Filter parameters
    age_group = request.args.get('age_group')
    gender = request.args.get('gender')
    province = request.args.get('province')
    ai_preference = request.args.get('ai_preference')

    filtered_data = data

    if age_group:
        filtered_data = [r for r in filtered_data if r.get('AgeGroup_Broad') == age_group]
    if gender:
        filtered_data = [r for r in filtered_data if r.get('Gender') == gender]
    if province:
        filtered_data = [r for r in filtered_data if r.get('Province') == province]
    if ai_preference:
        filtered_data = [r for r in filtered_data if r.get('Q10_Songs_by_AI') == ai_preference]

    # Return simplified version with key fields
    simplified = []
    for row in filtered_data:
        simplified.append({
            'participant_id': row['participant_id'],
            'age': row.get('Age', 'N/A'),
            'gender': row.get('Gender', 'N/A'),
            'province': row.get('Province', 'N/A'),
            'relationship_with_music': row.get('Q1_Relationship_with_music', 'N/A'),
            'discovering_music': row.get('Q2_Discovering_music', 'N/A'),
            'favorite_artist': row.get('Q3_artist_that_pulled_you_in', 'N/A'),
            'format_change': row.get('Q4_Music_format_changes', 'N/A'),
            'current_preference': row.get('Q9_Music_preference_these_days', 'N/A'),
            'ai_songs': row.get('Q10_Songs_by_AI', 'N/A'),
            'dead_artists_voice': row.get('Q11_Use_of_dead_artists_voice_feelings', 'N/A'),
            'theme_song': row.get('Q18_Life_theme_song', 'N/A'),
            'favorite_lyric': row.get('Q19_Lyric_that_stuck_with_you', 'N/A')
        })

    return jsonify({
        'total': len(simplified),
        'responses': simplified[:100]  # Limit to 100 for performance
    })

@app.route('/api/response/<participant_id>')
def get_response_detail(participant_id):
    """Get detailed view of a single response"""
    data = load_survey_data()

    for row in data:
        if row['participant_id'] == participant_id:
            return jsonify({
                'participant_id': row['participant_id'],
                'demographics': {
                    'age': row.get('Age', 'N/A'),
                    'gender': row.get('Gender', 'N/A'),
                    'province': row.get('Province', 'N/A'),
                    'education': row.get('Education', 'N/A'),
                    'income': row.get('HH_Income_Fine_23', 'N/A')
                },
                'music_relationship': {
                    'relationship': row.get('Q1_Relationship_with_music', 'N/A'),
                    'discovery_method': row.get('Q2_Discovering_music', 'N/A'),
                    'favorite_artist': row.get('Q3_artist_that_pulled_you_in', 'N/A'),
                    'format_change': row.get('Q4_Music_format_changes', 'N/A'),
                    'format_impact': row.get('Q5_Music_formal_change_impact', 'N/A'),
                    'current_preference': row.get('Q9_Music_preference_these_days', 'N/A')
                },
                'ai_opinions': {
                    'ai_songs': row.get('Q10_Songs_by_AI', 'N/A'),
                    'dead_artists_voice': row.get('Q11_Use_of_dead_artists_voice_feelings', 'N/A')
                },
                'personal': {
                    'theme_song': row.get('Q18_Life_theme_song', 'N/A'),
                    'favorite_lyric': row.get('Q19_Lyric_that_stuck_with_you', 'N/A'),
                    'guilty_pleasure': row.get('Q16_Music_guilty_pleasure_text_OE', 'N/A')
                }
            })

    return jsonify({'error': 'Participant not found'}), 404



if __name__ == '__main__':
    app.run(debug=True, port=5000)
