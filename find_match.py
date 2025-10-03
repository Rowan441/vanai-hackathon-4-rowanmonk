import os
import json
import numpy as np
from openai import OpenAI
import sys

# Base directory configuration
BASE_DIR = os.path.dirname(__file__)

from helpers.identity_string_utils import create_user_identity_string

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

QUESTIONS = [
    {
        "id": 1,
        "question": "What's your relationship with music like?",
        "placeholder": "e.g., 'It's my whole life' or 'I'm pretty casual about it'",
        "hint": "Just a sentence or two"
    },
    {
        "id": 2,
        "question": "How did you first discover music you loved?",
        "placeholder": "e.g., 'My dad's vinyl collection' or 'TikTok algorithm'",
        "hint": "What was the gateway?"
    },
    {
        "id": 3,
        "question": "What kind of music are you into these days?",
        "placeholder": "e.g., 'Sad girl indie' or 'Anything I can dance to'",
        "hint": "Vibe, genre, mood - whatever"
    },
    {
        "id": 4,
        "question": "Real talk - how do you feel about AI making music?",
        "placeholder": "e.g., 'If it's good, it's good' or 'Nah, needs human soul'",
        "hint": "No judgment, just curious"
    },
    {
        "id": 5,
        "question": "What about AI using dead artists' voices to make new songs?",
        "placeholder": "e.g., 'Kinda cool' or 'Feels wrong to me'",
        "hint": "First reaction is fine"
    },
    {
        "id": 6,
        "question": "Do you share music with people, or keep it to yourself?",
        "placeholder": "e.g., 'Always sending songs' or 'It's my private thing'",
        "hint": "How social are you with your taste?"
    }
]


def ask_questions():
    """Ask user the questions and collect answers"""
    print("\n🎵 Music Taste Matcher 🎵\n")
    print("Answer a few quick questions to find your music taste twin!\n")

    answers = {}

    for q in QUESTIONS:
        print(f"\nQuestion {q['id']}/6:")
        print(f"{q['question']}")
        print(f"💡 {q['hint']}")
        print(f"Example: {q['placeholder']}")

        answer = input("\nYour answer: ").strip()

        while not answer:
            print("Please provide an answer!")
            answer = input("Your answer: ").strip()

        answers[f"q{q['id']}"] = answer

    return answers


def get_embedding(text):
    """Get OpenAI embedding for text"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)


def load_survey_embeddings():
    """Load pre-computed survey embeddings from disk"""
    embeddings_file = os.path.join(BASE_DIR, "data/processed/survey_embeddings.json")

    if not os.path.exists(embeddings_file):
        print(f"Error: {embeddings_file} not found!")
        print("Please run 'python scripts/04_generate_embeddings.py' first to create the embeddings.")
        exit(1)

    print("\nLoading survey embeddings...")
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert embeddings back to numpy arrays
    for item in data:
        item['embedding'] = np.array(item['embedding'])

    return data


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_closest_match(n=5):
    """Main function to find top n closest survey matches"""
    # Get user responses
    answers = ask_questions()

    # Create identity string using the helper function
    print("\n\nGenerating your music taste profile...")
    user_identity_string = create_user_identity_string(answers)

    print("\nYour identity string:")
    print("-" * 60)
    print(user_identity_string)
    print("-" * 60)

    # Get embedding for user's identity string
    user_embedding = get_embedding(user_identity_string)

    # Load pre-computed survey embeddings
    survey_data = load_survey_embeddings()

    # Calculate similarities for all survey responses
    print(f"\nComparing with {len(survey_data)} survey responses...")
    matches = []

    for item in survey_data:
        similarity = cosine_similarity(user_embedding, item['embedding'])
        matches.append({
            'participant_id': item['participant_id'],
            'similarity_score': similarity,
            'match_data': item['data']
        })

    # Sort by similarity and get top n
    matches.sort(key=lambda x: x['similarity_score'], reverse=True)
    top_matches = matches[:n]

    # Display summary of best match
    best_match = top_matches[0]
    print("\n" + "=" * 80)
    print("🎉 Found your music taste twin! 🎉")
    print("=" * 80)
    print(f"\nParticipant ID: {best_match['participant_id']}")
    print(f"Similarity Score: {best_match['similarity_score']:.2%}\n")

    print("--- Their Music Profile ---")
    print(f"Relationship with music: {best_match['match_data'].get('Q1_Relationship_with_music', 'N/A')}")
    print(f"How they discover music: {best_match['match_data'].get('Q2_Discovering_music', 'N/A')}")
    print(f"Artist that pulled them in: {best_match['match_data'].get('Q3_artist_that_pulled_you_in', 'N/A')}")
    print(f"Music format change: {best_match['match_data'].get('Q4_Music_format_changes', 'N/A')}")
    print(f"Current preference: {best_match['match_data'].get('Q9_Music_preference_these_days', 'N/A')}")
    print(f"\n--- AI Opinions ---")
    print(f"Songs by AI: {best_match['match_data'].get('Q10_Songs_by_AI', 'N/A')}")
    print(f"Dead artists' voice feelings: {best_match['match_data'].get('Q11_Use_of_dead_artists_voice_feelings', 'N/A')}")
    print(f"\n--- Demographics ---")
    print(f"Age: {best_match['match_data'].get('Age', 'N/A')}")
    print(f"Gender: {best_match['match_data'].get('Gender', 'N/A')}")
    print(f"Location: {best_match['match_data'].get('Province', 'N/A')}")
    print("=" * 80)

    # Print summary of other top matches
    if n > 1:
        print(f"\n--- Other Top {n-1} Matches ---")
        for i, match in enumerate(top_matches[1:], 2):
            print(f"\n#{i} - Participant {match['participant_id']} ({match['similarity_score']:.2%} match)")
            print(f"  • {match['match_data'].get('Q1_Relationship_with_music', 'N/A')}")
            print(f"  • {match['match_data'].get('Q9_Music_preference_these_days', 'N/A')}")
            print(f"  • AI views: {match['match_data'].get('Q10_Songs_by_AI', 'N/A')}")

    return top_matches


if __name__ == "__main__":
    top_matches = find_closest_match(n=5)
    print(f"\n\nReturned {len(top_matches)} top matches")
    print(top_matches)