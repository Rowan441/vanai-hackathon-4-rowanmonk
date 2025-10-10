import json
import re

def highlight_entities_html(text, entities_json):
    """
    Convert annotated text with extracted entities to HTML with highlighting and Spotify links

    Args:
        text: Original text string (fallback if no annotated text)
        entities_json: JSON string containing annotated text and extracted entities

    Returns:
        HTML string with highlighted entities
    """
    if not entities_json:
        return text

    try:
        data = json.loads(entities_json) if isinstance(entities_json, str) else entities_json

        # Get annotated text (has inline markers)
        annotated_text = data.get('annotated_text', text)
        entities = data.get('entities', [])

        if not entities:
            return text

        # Replace inline markers with HTML
        # Pattern: ||{"type": "song", ...}text||
        def replace_marker(match):
            try:
                metadata_str = match.group(1)
                entity_text = match.group(2)
                metadata = json.loads(metadata_str)

                entity_type = metadata.get('type', 'unknown')
                css_class = f"music-entity music-entity-{entity_type}"

                # Find matching entity with Spotify link
                spotify_url = None
                for entity in entities:
                    if entity.get('matched_text') == entity_text:
                        spotify_url = entity.get('spotify_url')
                        break

                if spotify_url:
                    return f'<a href="{spotify_url}" target="_blank" class="{css_class}" title="Listen on Spotify">{entity_text} 🎵</a>'
                else:
                    return f'<span class="{css_class}">{entity_text}</span>'

            except:
                return match.group(2)  # Return just the text if parsing fails

        pattern = r'\|\|(\{[^}]+\})([^|]+)\|\|'
        result = re.sub(pattern, replace_marker, annotated_text)

        return result

    except Exception as e:
        print(f"Error highlighting entities: {e}")
        return text

def get_entity_summary(entities_json):
    """
    Get a summary of extracted entities

    Returns:
        Dict with counts and list of entities
    """
    if not entities_json:
        return None

    try:
        data = json.loads(entities_json) if isinstance(entities_json, str) else entities_json
        entities = data.get('entities', [])

        summary = {
            'total': len(entities),
            'songs': [],
            'artists': [],
            'albums': []
        }

        for entity in entities:
            entity_type = entity.get('type')
            entity_name = entity.get('name')

            if entity_type == 'song':
                song_info = {'name': entity_name}
                if 'artist' in entity:
                    song_info['artist'] = entity['artist']
                if 'spotify_url' in entity:
                    song_info['spotify_url'] = entity['spotify_url']
                summary['songs'].append(song_info)

            elif entity_type == 'artist':
                artist_info = {'name': entity_name}
                if 'spotify_url' in entity:
                    artist_info['spotify_url'] = entity['spotify_url']
                summary['artists'].append(artist_info)

            elif entity_type == 'album':
                summary['albums'].append({'name': entity_name})

        return summary

    except Exception as e:
        print(f"Error getting entity summary: {e}")
        return None
