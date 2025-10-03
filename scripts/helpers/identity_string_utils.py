import pandas as pd
import numpy as np

def create_survey_identity_string(row):
    """
    Survey respondent identity string
    Minimal processing - embeddings handle semantic matching
    """
    
    # Build string with natural language structure
    parts = []
    
    # Core identity
    parts.append(f"Music relationship: {row['Q1_Relationship_with_music']}")
    
    # Discovery background
    parts.append(f"First discovered music through: {row['Q2_Discovering_music']}")
    if pd.notna(row['Q3_artist_that_pulled_you_in']):
        parts.append(f"First artist that pulled them in: {row['Q3_artist_that_pulled_you_in']}")
    
    # # Format changes (shows adaptability)
    # if pd.notna(row['Q4_Music_format_changes']):
    #     parts.append(f"Most memorable format change: {row['Q4_Music_format_changes']}")
    # if pd.notna(row['Q6_Music_format_change_feelings']):
    #     parts.append(f"Felt about format changes: {row['Q6_Music_format_change_feelings']}")
    
    # Current behavior
    parts.append(f"Current music preference: {row['Q9_Music_preference_these_days']}")
    
    # Current discovery methods
    discovery_methods = []
    if pd.notna(row['Q7_New_music_discover_1']): discovery_methods.append("TikTok/Reels")
    if pd.notna(row['Q7_New_music_discover_2']): discovery_methods.append("streaming playlists")
    if pd.notna(row['Q7_New_music_discover_3']): discovery_methods.append("friend recommendations")
    if pd.notna(row['Q7_New_music_discover_4']): discovery_methods.append("movie/TV soundtracks")
    if pd.notna(row['Q7_New_music_discover_5']): discovery_methods.append("Shazam")
    if pd.notna(row['Q7_New_music_discover_6']): discovery_methods.append("music blogs")
    if pd.notna(row['Q7_New_music_discover_7']): discovery_methods.append("replays favorites")
    if discovery_methods:
        parts.append(f"Discovers new music through: {', '.join(discovery_methods)}")
    
    # AI attitudes (CRITICAL)
    parts.append(f"View on AI-generated music: {row['Q10_Songs_by_AI']}")
    parts.append(f"View on AI using dead artists' voices: {row['Q11_Use_of_dead_artists_voice_feelings']}")
    
    # Listening frequency (intensity signal)
    listening_contexts = []
    listen_map = {
        'Q8_Music_listen_time_GRID_1': 'waking up',
        'Q8_Music_listen_time_GRID_2': 'commuting',
        'Q8_Music_listen_time_GRID_3': 'working out',
        'Q8_Music_listen_time_GRID_4': 'cooking',
        'Q8_Music_listen_time_GRID_5': 'cleaning',
        'Q8_Music_listen_time_GRID_6': 'unwinding'
    }
    for field, context in listen_map.items():
        if pd.notna(row[field]) and row[field] in ['Often', 'Always']:
            listening_contexts.append(context)
    if listening_contexts:
        parts.append(f"Listens to music often/always when: {', '.join(listening_contexts)}")
    
    # Engagement behaviors
    behaviors = []
    if pd.notna(row['Q12_Music_bingo_1']): behaviors.append("makes breakup playlists")
    if pd.notna(row['Q12_Music_bingo_2']): behaviors.append("plays DJ on road trips")
    if pd.notna(row['Q12_Music_bingo_3']): behaviors.append("uses music for motivation")
    if pd.notna(row['Q12_Music_bingo_4']): behaviors.append("cries to sad songs")
    if pd.notna(row['Q12_Music_bingo_5']): behaviors.append("shares songs romantically")
    if pd.notna(row['Q12_Music_bingo_6']): behaviors.append("makes vibe playlists")
    if pd.notna(row['Q12_Music_bingo_7']): behaviors.append("replays same song many times")
    if behaviors:
        parts.append(f"Music behaviors: {', '.join(behaviors)}")
    
    # Sharing behavior (social dimension)
    sharing_methods = []
    if pd.notna(row['Q13_Share_the_music_you_love_1']): sharing_methods.append("texts links")
    if pd.notna(row['Q13_Share_the_music_you_love_2']): sharing_methods.append("group chats")
    if pd.notna(row['Q13_Share_the_music_you_love_3']): sharing_methods.append("social media")
    if pd.notna(row['Q13_Share_the_music_you_love_4']): sharing_methods.append("curates playlists")
    if pd.notna(row['Q13_Share_the_music_you_love_5']): sharing_methods.append("in-person")
    
    if pd.notna(row['Q13_Share_the_music_you_love_6']):
        parts.append("Doesn't share music")
    elif sharing_methods:
        parts.append(f"Shares music by: {', '.join(sharing_methods)}")
    
    parts.append(f"When friend shares music: {row['Q14_Friend_shares_a_song']}")
    
    # Self-perception
    parts.append(f"Guilty pleasure attitude: {row['Q15_Music_guilty_pleasure']}")
    if pd.notna(row['Q16_Music_guilty_pleasure_text_OE']):
        parts.append(f"Guilty pleasure song: {row['Q16_Music_guilty_pleasure_text_OE']}")
    
    # Genre (extracted feature)
    if pd.notna(row['extracted_genre']):
        parts.append(f"Genre preference: {row['extracted_genre']}")
    
    # Favorite band (extracted feature)
    if pd.notna(row['extracted_favourite_band']):
        parts.append(f"Favorite artist: {row['extracted_favourite_band']}")
    
    return "\n".join(parts)


def create_user_identity_string(answers):
    """
    User quiz identity string
    Same natural language structure as survey
    """
    
    parts = []
    
    # Core identity
    parts.append(f"Music relationship: {answers['q1']}")
    
    # Discovery background
    parts.append(f"First discovered music through: {answers['q2']}")
    
    # Current behavior
    parts.append(f"Current music preference: {answers['q3']}")
    
    # AI attitudes (CRITICAL)
    parts.append(f"View on AI-generated music: {answers['q4']}")
    parts.append(f"View on AI using dead artists' voices: {answers['q5']}")
    
    # Sharing behavior (social dimension)
    parts.append(f"Music sharing behavior: {answers['q6']}")
    
    return "\n".join(parts)