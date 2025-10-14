# Music Taste Matcher

**Find Your Sonic Twin**

A Van AI Hackathon Round 4 Project

## Team Members

- Rowan Monk

## Project Overview

Music Taste Matcher is an AI-powered web application that connects people through their unique relationship with music. We surveyed hundreds of real people about their musical journey from their first beloved artist to their guilty pleasures, from how they discover new music to the lyrics that changed them. The result is a rich dataset of authentic music stories waiting to be discovered.

When users answer six thoughtfully crafted questions about their own musical identity, our application uses OpenAI's text embedding models to perform semantic matching, finding the survey respondent whose musical DNA most closely mirrors theirs. This isn't just about matching favorite genres or artists it's about matching the deeper relationship people have with music: how they discover it, when they listen, how intensely they engage, and how they share it with others.

The experience goes beyond the match itself. Users discover their twin's complete music profile, enriched with Spotify links to explore their favorite artists and songs directly. Our AI analysis provides personalized insights explaining why they were paired together, highlighting genuine connections in their responses. To visualize these musical identities, we generate custom illustrated avatars for both the user and their match, with artistic styles that reflect their attitudes toward AI, music intensity levels, and social sharing behaviors.

Music Taste Matcher transforms abstract survey data into meaningful human connections, proving that our relationship with music is as unique as our fingerprints and just as recognizable when you know what to look for.

---

## Technical Implementation

### Architecture & Stack

The application is built with a Flask backend and vanilla JavaScript frontend, prioritizing rapid development and straightforward deployment. The backend handles API orchestration, embeddings generation, and real-time avatar creation, while the frontend provides an engaging, progressive questionnaire experience with smooth animations and responsive design.

### AI & Matching System

At the core is OpenAI's `text-embedding-3-small` model, which converts user responses into high-dimensional vector representations. We pre-computed embeddings for all survey respondents, enabling real-time matching via cosine similarity search. The identity string construction intelligently synthesizes user responses into a coherent narrative that captures both explicit answers and implicit preferences.

Our avatar generation system uses DALL-E (gpt-image-1) with dynamically constructed prompts. We map survey responses to three key dimensions: AI attitude spectrum (embracer to rejector), music intensity (obsessed to minimal), and social sharing behavior (active curator to hoarder). Each dimension influences different aspects of the generated avatar aesthetic style and lighting for AI attitude, expression and accessories for intensity, and background elements for sociality. Prompt variations are randomized within categories to ensure visual diversity while maintaining thematic consistency.

### Data Pipeline & Entity Extraction

The data processing pipeline consists of multiple scripts that progressively enrich the survey data. We use GPT-4o-mini for music entity extraction, identifying songs, artists, and albums in free-text responses through inline annotation with JSON metadata. The extraction system includes robust validation with retry logic, ensuring text fidelity while enabling Spotify URL enrichment. Entity annotations are converted to clickable Spotify links in the UI, allowing users to immediately explore their match's musical taste.

The matching insights feature employs a secondary LLM call that compares user responses with their match's full profile, generating 0-3 high-quality insights that reference specific answers as evidence. This prevents generic observations and ensures meaningful connections.

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- Environment Variables:
  - `OPENAI_API_KEY`
  - `SPOTIFY_CLIENT_ID`
  - `SPOTIFY_CLIENT_SECRET`


### Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install dependencies:
```bash
pip install -r src/requirements.txt
```

3. Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export SPOTIFY_CLIENT_ID="your-spotify-client-id"
export SPOTIFY_CLIENT_SECRET="your-spotify-client-secret"
```

4. Run the Flask application:
```bash
cd src
python app.py
```

5. Open your browser to `http://localhost:5000`

### Data Processing Pipeline (Optional)

To regenerate survey embeddings and entity extractions from original data: `data\raw\music_survey_data.csv`:

```bash
cd scripts
python 01_clean_data.py
python 02_extract_genre_bands.py
python 04_generate_survey_embeddings.py
python 05_extract_music_entities.py
python 06_extract_favourite_artist.py
```