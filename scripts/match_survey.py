import os
import json
import numpy as np
from openai import OpenAI

# Base directory configuration
BASE_DIR = os.path.dirname(__file__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def ask_test_questions():
    """Ask user a short test about their music and AI preferences"""
    print("\n🎵 Music & AI Taste Test 🎵\n")

    questions = [
        "If your phone only had space for one app, would it be a music app?",
        "Tell me about the last song that got stuck in your head - how'd you find it?",
        "Who's an artist you could talk about for hours?",
        "Do you still have any old CDs, cassettes, or vinyl lying around? Why or why not?",
        "What kind of vibe are you looking for when you hit play lately?",
        "If a song slaps but it's made by AI, does it still slap?",
        "Imagine Tupac or Kurt Cobain dropping a 'new' AI-generated track tomorrow. What's your gut reaction?"
    ]

    answers = []
    for i, question in enumerate(questions, 1):
        answer = input(f"{i}. {question}\n   Your answer: ")
        answers.append(answer)

    # Create a combined text representation
    combined_text = " | ".join([f"Q{i+1}: {ans}" for i, ans in enumerate(answers)])
    return combined_text

def get_embedding(text):
    """Get OpenAI embedding for text"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

def load_survey_embeddings():
    """Load pre-computed survey embeddings from disk"""
    embeddings_file = os.path.join(BASE_DIR, "../data/processed/survey_embeddings.json")

    if not os.path.exists(embeddings_file):
        print(f"Error: {embeddings_file} not found!")
        print("Please run 'python generate_embeddings.py' first to create the embeddings cache.")
        exit(1)

    print("Loading cached survey embeddings...")
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert embeddings back to numpy arrays
    for item in data:
        item['embedding'] = np.array(item['embedding'])

    return data

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_closest_match():
    """Main function to find closest survey match"""
    # Get user responses
    print("\nLet's find your music taste match from our survey!\n")
    user_text = ask_test_questions()

    print("\n\nGenerating your music taste profile...")
    user_embedding = get_embedding(user_text)

    # Load pre-computed survey embeddings
    survey_data = load_survey_embeddings()

    # Find best match
    print(f"Comparing with {len(survey_data)} survey responses...")
    best_similarity = -1
    best_match = None

    for item in survey_data:
        similarity = cosine_similarity(user_embedding, item['embedding'])

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = item['data']

    # Display results
    print("\n" + "="*60)
    print("🎉 Found your music taste twin! 🎉")
    print("="*60)
    print(f"\nSimilarity Score: {best_similarity:.2%}\n")
    print(f"Their relationship with music: {best_match['Q1_Relationship_with_music']}")
    print(f"How they discover music: {best_match['Q2_Discovering_music']}")
    print(f"Artist that pulled them in: {best_match['Q3_artist_that_pulled_you_in']}")
    print(f"Music format change: {best_match['Q4_Music_format_changes']}")
    print(f"Current preference: {best_match['Q9_Music_preference_these_days']}")
    print(f"Songs by AI: {best_match['Q10_Songs_by_AI']}")
    print(f"Dead artists' voice feelings: {best_match['Q11_Use_of_dead_artists_voice_feelings']}")
    print(f"\nAge: {best_match['Age']}")
    print(f"Gender: {best_match['Gender']}")
    print(f"Location: {best_match['Province']}")
    print("="*60)

if __name__ == "__main__":
    find_closest_match()
