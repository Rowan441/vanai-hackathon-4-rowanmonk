from flask import Flask, render_template, jsonify, request
import json
import csv
import os
import sys
import numpy as np
from collections import Counter
from openai import OpenAI

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from helpers.identity_string_utils import create_user_identity_string
from models import RespondentProfile, MatchResult, QuestionnaireResponse

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Base directory configuration
BASE_DIR = os.path.dirname(__file__)

app = Flask(__name__)

def load_survey_data():
    """Load survey data from CSV"""
    survey_file = os.path.join(BASE_DIR, "../data/processed/02_music_survey_with_genres.csv")
    with open(survey_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

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
        print("Received answers:", data)

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

        profile = RespondentProfile(
            age=matched_response.get('Age', 'N/A'),
            gender=matched_response.get('Gender', 'N/A'),
            province=matched_response.get('Province', 'N/A'),
            relationship_with_music=matched_response.get('Q1_Relationship_with_music', 'N/A'),
            discovering_music=matched_response.get('Q2_Discovering_music', 'N/A'),
            favorite_artist=matched_response.get('Q3_artist_that_pulled_you_in', 'N/A'),
            current_preference=matched_response.get('Q9_Music_preference_these_days', 'N/A'),
            ai_songs=matched_response.get('Q10_Songs_by_AI', 'N/A'),
            dead_artists_voice=matched_response.get('Q11_Use_of_dead_artists_voice_feelings', 'N/A'),
            theme_song=matched_response.get('Q18_Life_theme_song', 'N/A'),
            favorite_lyric=matched_response.get('Q19_Lyric_that_stuck_with_you', 'N/A'),
            genre=matched_response.get('extracted_genre', 'N/A'),
            favorite_band=matched_response.get('extracted_favourite_band', 'N/A')
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
