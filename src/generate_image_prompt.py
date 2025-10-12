import pandas as pd
import numpy as np
from enum import Enum

class AISpectrumLevel(Enum):
    """AI attitude levels"""
    EMBRACER = 3
    CURIOUS = 2
    UNCERTAIN = 1
    REJECTOR = 0

class IntensityLevel(Enum):
    """Music intensity levels"""
    OBSESSED = 3
    ENGAGED = 2
    CASUAL = 1
    MINIMAL = 0

class SocialityLevel(Enum):
    """Music sociality levels"""
    HOARDER = 0
    CASUAL_SHARER = 1
    SOCIAL_LISTENER = 2
    ACTIVE_CURATOR = 3

class MusicGenre(Enum):
    """Predefined music genres for classification."""
    ROCK = "rock"
    POP = "pop"
    HIP_HOP = "hip hop"
    RAP = "rap"
    RNB = "r&b"
    COUNTRY = "country"
    INDIE = "indie"
    ALTERNATIVE = "alternative"
    METAL = "metal"
    PUNK = "punk"
    ELECTRONIC = "electronic"
    EDM = "edm"
    JAZZ = "jazz"
    BLUES = "blues"
    FOLK = "folk"
    CLASSICAL = "classical"
    REGGAE = "reggae"
    LATIN = "latin"
    KPOP = "k-pop"
    OTHER = "other"

def create_image_prompt_from_survey(data):
    """
    Generate musical avatar image prompt from survey data
    """

    # Calculate interpreted dimensions
    ai_embracer_level = calculate_ai_spectrum(data)  # Returns (enum, score)
    intensity_level = calculate_intensity(data)  # Returns (enum, score)
    sociality_level = calculate_sociality(data)  # Returns (enum, score)
    favourite_band = data.get("extracted_favourite_band")
    favourite_genre = data.get("extracted_genre")

    # Generate Physical description
    physical_desc = generate_physical_description(
        ethnicity=data.get('Ethnicity_Roll_23'),
        age=data.get('Age'),
        gender=data.get('Gender')
    )

    # Generate Musical Avatar Image Prompt
    avatar_prompt = generate_avatar_prompt(
        physical_desc=physical_desc,
        ai_level=ai_embracer_level[0],
        intensity_level=intensity_level[0],
        sociality_level=sociality_level[0],
        favourite_genre=favourite_genre,
        favourite_band=favourite_band
    )

    return avatar_prompt


def generate_physical_description(ethnicity, age, gender):
    """Generate physical description from demographics"""
    parts = []

    # Age descriptor
    if pd.notna(age):
        age_int = int(age)
        if age_int < 25:
            parts.append("young adult")
        elif age_int < 35:
            parts.append("person in their late 20s-early 30s")
        elif age_int < 45:
            parts.append("person in their late 30s-early 40s")
        elif age_int < 55:
            parts.append("middle-aged person")
        else:
            parts.append("mature adult")

    # Gender
    if pd.notna(gender):
        gender_str = str(gender).lower()
        if 'male' in gender_str and 'female' not in gender_str:
            parts.append("man")
        elif 'female' in gender_str:
            parts.append("woman")
        else:
            parts.append("person")
    else:
        parts.append("person")

    # Ethnicity
    if pd.notna(ethnicity):
        ethnicity_str = str(ethnicity).strip()
        if ethnicity_str == "White":
            parts.append("of European descent")
        elif ethnicity_str == "Chinese":
            parts.append("of Chinese descent")
        elif ethnicity_str == "South Asian":
            parts.append("of South Asian descent")
        elif ethnicity_str == "First Nations":
            parts.append("of Indigenous descent")
        elif ethnicity_str == "Black":
            parts.append("of African descent")
        elif ethnicity_str == "Arab/West Asian":
            parts.append("of Middle Eastern descent")
        elif ethnicity_str == "East Asian":
            parts.append("of East Asian descent")
        elif ethnicity_str == "Latin American":
            parts.append("of Latin American descent")
        elif ethnicity_str == "Filipino":
            parts.append("of Filipino descent")
        elif ethnicity_str == "Southeast Asian":
            parts.append("of Southeast Asian descent")
        elif ethnicity_str == "Multiple visible minorities":
            parts.append("of mixed ethnic background")

    return ", ".join(parts) if parts else "person"


def generate_avatar_prompt(physical_desc, ai_level, intensity_level, sociality_level, favourite_genre, favourite_band):
    """Generate image prompt for musical avatar"""

    prompt = f"Avatar of a {physical_desc}, "

    wearing = "wearing "
    # Band t-shirt - REQUIRED if band is known
    if pd.notna(favourite_band) and str(favourite_band).lower() not in ['unknown', 'unknown artist', 'nan']:
        wearing += f"a {favourite_band} t-shirt, "

    if pd.notna(favourite_genre):
        genre_lower = str(favourite_genre).lower()
        if "other" not in genre_lower:
            wearing += genre_lower + " music fashion. "
        else:
            wearing += "casual music fan attire. "
    else:
        wearing += "casual music fan attire. "

    prompt += wearing

    # Intensity influences energy and expression
    if intensity_level == IntensityLevel.OBSESSED:
        prompt += "Passionate intense expression, lots of music accessories. "
    elif intensity_level == IntensityLevel.ENGAGED:
        prompt += "Enthusiastic expression, vibing with the music. "
    elif intensity_level == IntensityLevel.CASUAL:
        prompt += "Relaxed expression. Casually enjoying the music. "
    else:
        prompt += "Understated expression. "

    # # Sociality influences background/setting - number of people
    # if sociality_level == SocialityLevel.ACTIVE_CURATOR:
    #     prompt += "Background: Crowd of friends around them. "
    # elif sociality_level == SocialityLevel.SOCIAL_LISTENER:
    #     prompt += "Background: With 2-3 close friends, intimate music sharing moment. "
    # elif sociality_level == SocialityLevel.CASUAL_SHARER:
    #     prompt += "Background: One friend in the background, casual setting. "
    # else:
    #     prompt += "Background: Alone "

    # AI attitude influences aesthetic style/lighting
    if ai_level == AISpectrumLevel.EMBRACER:
        prompt += "Futuristic digital aesthetic. "
    elif ai_level == AISpectrumLevel.CURIOUS or ai_level == AISpectrumLevel.UNCERTAIN:
        prompt += "Modern aesthetic. "
    else:
        prompt += "Classic vintage aesthetic."

    prompt += "detailed digital illustration, music-themed."

    return prompt

def calculate_ai_spectrum(data):
    """Map answers to AI attitude spectrum"""
    q10_map = {
        "Yes – and I already have": AISpectrumLevel.EMBRACER,
        "Sure I would — if it sounds good, why not?": AISpectrumLevel.CURIOUS,
        "Maybe — I’m curious": AISpectrumLevel.CURIOUS,
        "Not sure yet": AISpectrumLevel.UNCERTAIN,
        "Nah — I prefer music made by real people": AISpectrumLevel.REJECTOR,
    }

    q11_map = {
        "I’m into it — it keeps their legacy alive": AISpectrumLevel.EMBRACER,
        "I’m unsure — it depends how it’s done": AISpectrumLevel.UNCERTAIN,
        "Hadn’t thought about it before": AISpectrumLevel.UNCERTAIN,
        "I don’t like it — it feels wrong": AISpectrumLevel.REJECTOR,
    }


    q10_answer = q10_map.get(data['Q10_Songs_by_AI'])
    q11_answer = q11_map.get(data['Q11_Use_of_dead_artists_voice_feelings'])
    
    if (q10_answer == AISpectrumLevel.EMBRACER):
        ai_level = AISpectrumLevel.EMBRACER
    else:
        ai_level = AISpectrumLevel(max(q10_answer.value, q11_answer.value))
    return ai_level, ai_level.value/3.0

def calculate_intensity(data):
    """
    Calculate music intensity from multiple signals
    """
    
    # Base intensity from Q1 (self-reported)
    q1_map = {
        "I’m obsessed 🎵": 4,
        "I like it, but don’t keep up": 2,
        "I’m more of a casual listener": 1,
        "Meh — it’s not a big part of my life": 0
    }
    base_score = q1_map.get(data['Q1_Relationship_with_music'], 2)
    
    # Behavioral frequency from Q8 (when they listen)
    # Sum up: Never=0, Sometimes=1, Often=2, Always=3
    q8_score = 0
    q8_fields = [
        'Q8_Music_listen_time_GRID_1',  # Waking up
        'Q8_Music_listen_time_GRID_2',  # Commuting
        'Q8_Music_listen_time_GRID_3',  # Working out
        'Q8_Music_listen_time_GRID_4',  # Cooking
        'Q8_Music_listen_time_GRID_5',  # Cleaning
        'Q8_Music_listen_time_GRID_6',  # Unwinding
    ]
    
    frequency_map = {
        'Never': 0,
        'Sometimes': 1,
        'Often': 2,
        'Always': 3
    }
    
    for field in q8_fields:
        if field in data:
            q8_score += frequency_map.get(data[field], 0)
    
    # Normalize Q8 to 0-4 scale (max possible is 18)
    q8_normalized = (q8_score / 18) * 4
    
    # Engagement depth from Q12 (music bingo)
    # Count how many boxes they checked
    q12_fields = [
        'Q12_Music_bingo_1',  # breakup playlist
        'Q12_Music_bingo_2',  # DJ on road trip
        'Q12_Music_bingo_3',  # hype music
        'Q12_Music_bingo_4',  # cried to song
        # 'Q12_Music_bingo_5',  # flirting with songs
        'Q12_Music_bingo_6',  # vibes playlist
        'Q12_Music_bingo_7',  # replayed same song 10+ times
    ]
    
    q12_count = sum(1 for field in q12_fields if field in data and pd.notna(data[field]))
    
    # Normalize Q12 to 0-4 scale (max is 6)
    q12_normalized = (q12_count / 6) * 4
    
    # Weighted average: Q1 most important, then Q8, then Q12
    final_score = (
        base_score * 0.4 +      # Self-report: 50%
        q8_normalized * 0.35 +    # Frequency: 30%
        q12_normalized * 0.25     # Engagement: 20%
    )

    # Normalize to 0-1 (max score: 4)
    normalized_score = final_score / 4.0

    # Map to categories
    if normalized_score >= 3.5/4.0:
        return IntensityLevel.OBSESSED, normalized_score
    elif normalized_score >= 2.5/4.0:
        return IntensityLevel.ENGAGED, normalized_score
    elif normalized_score >= 1.5/4.0:
        return IntensityLevel.CASUAL, normalized_score
    else:
        return IntensityLevel.MINIMAL, normalized_score

def calculate_sociality(data):
    """
    Calculate music sociality from sharing and discovery behaviors
    Music Hoarder ←→ Casual Sharer ←→ Active Curator
    """

    score = 0

    # Discovery through social channels (+1 each)
    if pd.notna(data.get('Q7_New_music_discover_3')):  # Friend recs
        score += 1
    if pd.notna(data.get('Q7_New_music_discover_7')):  # Music blogs or critics
        score += 1

    # Music bingo social behaviors (+1 each)
    if pd.notna(data.get('Q12_Music_bingo_2')):  # DJ on road trip
        score += 1
    if pd.notna(data.get('Q12_Music_bingo_5')):  # Shared a song to flirt
        score += 1

    # Sharing methods (count how many ways they share)
    sharing_count = 0
    if pd.notna(data.get('Q13_Share_the_music_you_love_1')):  # Texting links
        sharing_count += 1
    if pd.notna(data.get('Q13_Share_the_music_you_love_2')):  # Group chats
        sharing_count += 1
    if pd.notna(data.get('Q13_Share_the_music_you_love_3')):  # Social media
        sharing_count += 1
    if pd.notna(data.get('Q13_Share_the_music_you_love_4')):  # Curating playlists
        sharing_count += 1  # Weighted higher - more curation effort
    if pd.notna(data.get('Q13_Share_the_music_you_love_5')):  # In-person
        sharing_count += 1

    # Add sharing count to score
    score += sharing_count

    # Response to friend sharing music (+bonus points for engagement)
    q14_map = {
        "Listen right away": 2,
        "Save it for later": 1,
        "Depends on the friend!": 1,
        "Pretend I listened 😬": 0
    }
    score += q14_map.get(data.get('Q14_Friend_shares_a_song', ''), 0)

    # Normalize and categorize
    # Max possible: 2 (discovery) + 2 (bingo) + 5 (sharing) + 2 (friend response) = 11

    # Normalize to 0-1 (min score: 0, max score: 11)
    normalized_score = score / 11.0

    if normalized_score >= 6.0/11.0:
        return SocialityLevel.ACTIVE_CURATOR, normalized_score
    elif normalized_score >= (4.0/11.0):
        return SocialityLevel.SOCIAL_LISTENER, normalized_score
    elif normalized_score >= (2/11.0):
        return SocialityLevel.CASUAL_SHARER, normalized_score
    else:
        return SocialityLevel.HOARDER, normalized_score


