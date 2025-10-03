import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Base directory configuration
BASE_DIR = os.path.dirname(__file__)

# Add helpers to path
sys.path.insert(0, os.path.join(BASE_DIR, 'helpers'))
from generate_image_prompt import (
    calculate_ai_spectrum,
    calculate_intensity,
    calculate_sociality,
    AISpectrumLevel,
    IntensityLevel,
    SocialityLevel
)

def calculate_all_scores(sample_size=None):
    """Calculate personality scores for all survey respondents"""

    # Load the processed survey data
    input_csv = os.path.join(BASE_DIR, "../data/processed/music_survey_with_genres.csv")

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    if sample_size:
        df = df.sample(n=min(sample_size, len(df)))

    print(f"Calculating personality scores for {len(df)} respondents...")

    results = []

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(df)}...", end="\r")

        try:
            # Calculate all dimensions
            ai_level, ai_score = calculate_ai_spectrum(row)
            intensity_level, intensity_score = calculate_intensity(row)
            sociality_level, sociality_score = calculate_sociality(row)

            results.append({
                'participant_id': row['participant_id'],
                'ai_level': ai_level.name,
                'ai_score': ai_score,
                'intensity_level': intensity_level.name,
                'intensity_score': intensity_score,
                'sociality_level': sociality_level.name,
                'sociality_score': sociality_score
            })
        except Exception as e:
            print(f"\nError processing participant {row['participant_id']}: {e}")
            continue

    print(f"\n✓ Processed {len(results)} respondents")

    return pd.DataFrame(results)


def plot_personality_distributions(df):
    """Create linear scatterplot visualizations for each personality dimension"""

    # Set up the figure with 3 subplots (one for each dimension)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Music Personality Distribution', fontsize=16, fontweight='bold')

    # Color maps for each dimension
    ai_colors = {
        'EMBRACER': '#00ff88',      # Bright green
        'CURIOUS': '#4d94ff',       # Blue
        'UNCERTAIN': '#ffaa00',     # Orange
        'REJECTOR': '#ff4444'       # Red
    }

    intensity_colors = {
        'OBSESSED': '#ff00ff',      # Magenta
        'ENGAGED': '#ff6600',       # Orange
        'CASUAL': '#ffcc00',        # Yellow
        'MINIMAL': '#999999'        # Gray
    }

    sociality_colors = {
        'ACTIVE_CURATOR': '#00ffff',    # Cyan
        'SOCIAL_LISTENER': '#00cc66',   # Green
        'CASUAL_SHARER': '#ffaa44',     # Light orange
        'HOARDER': '#6633cc'            # Purple
    }

    # Plot 1: AI Spectrum - Bar Chart
    ax1 = axes[0]

    # Count occurrences of each level
    ai_counts = df['ai_level'].value_counts()

    # Ensure all levels are present (even if count is 0)
    level_order = ['REJECTOR', 'UNCERTAIN', 'CURIOUS', 'EMBRACER']
    counts = [ai_counts.get(level, 0) for level in level_order]
    colors_ordered = [ai_colors[level] for level in level_order]

    # Create bar chart
    bars = ax1.bar(level_order, counts, color=colors_ordered, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_xlabel('AI Attitude Level', fontsize=11)
    ax1.set_title('AI Attitude Distribution', fontsize=12, pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)

    # Plot 2: Intensity - Histogram
    ax2 = axes[1]

    # Create histogram with color-coded bins by level
    _, bins, patches = ax2.hist(df['intensity_score'], bins=[n/8 for n in range(9)], edgecolor='black',
                                 linewidth=1, alpha=0.8)

    # Color-code bins by intensity level thresholds
    # Thresholds based on calculate_intensity: >=3.5=Obsessed, >=2.5=Engaged, >=1.5=Casual, else=Minimal
    # Normalized: (score-1)/3 so: >=0.833=Obsessed, >=0.5=Engaged, >=0.167=Casual
    for i, patch in enumerate(patches):
        bin_right = bins[i+1]
        if bin_right > 3.5/4.0:
            patch.set_facecolor(intensity_colors['OBSESSED'])
        elif bin_right > 2.5/4.0:
            patch.set_facecolor(intensity_colors['ENGAGED'])
        elif bin_right > 1.5/4.0:
            patch.set_facecolor(intensity_colors['CASUAL'])
        else:
            patch.set_facecolor(intensity_colors['MINIMAL'])

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=intensity_colors['MINIMAL'], edgecolor='black', label='Minimal'),
        Patch(facecolor=intensity_colors['CASUAL'], edgecolor='black', label='Casual'),
        Patch(facecolor=intensity_colors['ENGAGED'], edgecolor='black', label='Engaged'),
        Patch(facecolor=intensity_colors['OBSESSED'], edgecolor='black', label='Obsessed')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=9)

    ax2.set_xlabel('Normalized Score (0 = Minimal, 1 = Obsessed)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Music Intensity Distribution', fontsize=12, pad=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)

    # Plot 3: Sociality - Histogram
    ax3 = axes[2]

    # Create histogram with color-coded bins by level
    _, bins, patches = ax3.hist(df['sociality_score'], bins=[n/11 for n in range(12)], edgecolor='black',
                                 linewidth=1, alpha=0.8)

    for i, patch in enumerate(patches):
        bin_right = bins[i+1]
        if bin_right <= 2/11.0:
            patch.set_facecolor(sociality_colors['HOARDER'])
        elif bin_right <= 4/11.0:
            patch.set_facecolor(sociality_colors['CASUAL_SHARER'])
        elif bin_right <= 6/11.0:
            patch.set_facecolor(sociality_colors['SOCIAL_LISTENER'])
        else:
            patch.set_facecolor(sociality_colors['ACTIVE_CURATOR'])

    # Create custom legend
    legend_elements = [
        Patch(facecolor=sociality_colors['HOARDER'], edgecolor='black', label='Hoarder'),
        Patch(facecolor=sociality_colors['CASUAL_SHARER'], edgecolor='black', label='Casual Sharer'),
        Patch(facecolor=sociality_colors['SOCIAL_LISTENER'], edgecolor='black', label='Social Listener'),
        Patch(facecolor=sociality_colors['ACTIVE_CURATOR'], edgecolor='black', label='Active Curator')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=9)

    ax3.set_xlabel('Normalized Score (0 = Hoarder, 1 = Active Curator)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Music Sociality Distribution', fontsize=12, pad=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_axisbelow(True)

    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(BASE_DIR, "../data/analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "personality_distributions.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")

    plt.show()

    return fig


def print_distribution_stats(df):
    """Print summary statistics for each dimension"""

    print("\n" + "="*80)
    print("DISTRIBUTION STATISTICS")
    print("="*80)

    # AI Spectrum
    print("\n--- AI Spectrum ---")
    print(df['ai_level'].value_counts().sort_index())
    print(f"Mean Score: {df['ai_score'].mean():.3f}")
    print(f"Std Dev: {df['ai_score'].std():.3f}")

    # Intensity
    print("\n--- Music Intensity ---")
    print(df['intensity_level'].value_counts().sort_index())
    print(f"Mean Score: {df['intensity_score'].mean():.3f}")
    print(f"Std Dev: {df['intensity_score'].std():.3f}")

    # Sociality
    print("\n--- Music Sociality ---")
    print(df['sociality_level'].value_counts().sort_index())
    print(f"Mean Score: {df['sociality_score'].mean():.3f}")
    print(f"Std Dev: {df['sociality_score'].std():.3f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Calculate scores for all respondents (or set sample_size for subset)
    SAMPLE_SIZE = None  # Set to None for all data, or a number for sample

    df = calculate_all_scores(sample_size=SAMPLE_SIZE)

    # Save to CSV
    output_dir = os.path.join(BASE_DIR, "../data/analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "personality_scores.csv")
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved scores to: {output_csv}")

    # Print statistics
    print_distribution_stats(df)

    # Create visualization
    plot_personality_distributions(df)
