from pydantic import BaseModel
from typing import Optional


class RespondentProfile(BaseModel):
    """Survey respondent profile data"""
    age: Optional[str] = "N/A"
    gender: Optional[str] = "N/A"
    province: Optional[str] = "N/A"
    relationship_with_music: Optional[str] = "N/A"
    discovering_music: Optional[str] = "N/A"
    favorite_artist: Optional[str] = "N/A"
    current_preference: Optional[str] = "N/A"
    ai_songs: Optional[str] = "N/A"
    dead_artists_voice: Optional[str] = "N/A"
    theme_song: Optional[str] = "N/A"
    favorite_lyric: Optional[str] = "N/A"
    genre: Optional[str] = "N/A"
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
