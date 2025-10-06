from pydantic import BaseModel
from typing import Optional


class RespondentProfile(BaseModel):
    """Survey respondent profile data"""
    age: Optional[str] = "N/A"
    gender: Optional[str] = "N/A"
    location: Optional[str] = "N/A"
    relationship_with_music: Optional[str] = "N/A"
    discovering_music: Optional[str] = "N/A"
    first_song_artist_love: Optional[str] = "N/A"
    format_change: Optional[str] = "N/A"
    format_change_memory: Optional[str] = "N/A"
    format_change_feelings: Optional[str] = "N/A"
    discovery_methods: Optional[str] = "N/A"
    listening_contexts: Optional[str] = "N/A"
    current_preference: Optional[str] = "N/A"
    ai_songs: Optional[str] = "N/A"
    dead_artists_voice: Optional[str] = "N/A"
    music_achievements: Optional[str] = "N/A"
    sharing_methods: Optional[str] = "N/A"
    friend_shares_reaction: Optional[str] = "N/A"
    guilty_pleasure_attitude: Optional[str] = "N/A"
    guilty_pleasure_song: Optional[str] = "N/A"
    theme_song: Optional[str] = "N/A"
    favorite_lyric: Optional[str] = "N/A"
    favorite_genre: Optional[str] = "N/A"
    favorite_band: Optional[str] = "N/A"


class MatchResult(BaseModel):
    """Match result with similarity score and profile"""
    participant_id: str
    similarity_score: float
    profile: RespondentProfile


class QuestionnaireResponse(BaseModel):
    """Response to questionnaire submission"""
    status: str
    match: MatchResult
