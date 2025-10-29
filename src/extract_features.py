"""
Extract acoustic features from audio files using OpenSMILE.

This module processes MP3 audio files and extracts eGeMAPSv02 acoustic features
at multiple time points throughout each song. Features are saved as CSV files.
"""

import os
import argparse
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
import audiofile
import opensmile


def set_df(df, group, location, target):
    """
    Add metadata columns to feature dataframe.

    Args:
        df: DataFrame containing extracted features
        group: Group identifier (e.g., genre_songindex)
        location: Sample location identifier
        target: Target class label

    Returns:
        DataFrame with added metadata columns
    """
    df['group'] = group
    df['token'] = location
    df['target'] = target
    return df


def extract_features(file, smile, group, target):
    """
    Extract acoustic features from multiple segments of an audio file.

    Extracts 10-second segments at 4 evenly-spaced positions throughout the song
    (at 1/3, 1/3+10s, 1/3+20s, and 1/3+30s of the song duration).

    Args:
        file: Path to audio file
        smile: OpenSMILE Smile object for feature extraction
        group: Group identifier for this audio file
        target: Target class label (0, 1, or 2)

    Returns:
        DataFrame containing features from all 4 segments
    """
    # Get length of audio
    signal, sampling_rate = audiofile.read(file, always_2d=True)
    length_seconds = signal.shape[1] / sampling_rate

    # Calculate sample positions (4 samples spaced around 1/3 point)
    sample_1 = length_seconds // 3 - 10
    sample_2 = length_seconds // 3
    sample_3 = length_seconds // 3 + 10
    sample_4 = length_seconds // 3 + 20

    # Ensure sample positions are valid
    if sample_1 < 0:
        sample_1 = 0

    # Read 10-second segments at different time points
    signal_1, _ = audiofile.read(file, duration=10, offset=sample_1, always_2d=True)
    signal_2, _ = audiofile.read(file, duration=10, offset=sample_2, always_2d=True)
    signal_3, _ = audiofile.read(file, duration=10, offset=sample_3, always_2d=True)
    signal_4, _ = audiofile.read(file, duration=10, offset=sample_4, always_2d=True)

    # Extract features from each segment
    df_1 = smile.process_signal(signal_1, sampling_rate)
    df_1 = set_df(df_1, group, '1', target)

    df_2 = smile.process_signal(signal_2, sampling_rate)
    df_2 = set_df(df_2, group, '2', target)

    df_3 = smile.process_signal(signal_3, sampling_rate)
    df_3 = set_df(df_3, group, '3', target)

    df_4 = smile.process_signal(signal_4, sampling_rate)
    df_4 = set_df(df_4, group, '4', target)

    return pd.concat([df_1, df_2, df_3, df_4])


def extract_concatenate_per_class(files_path, target, genre_name, smile):
    """
    Extract features from all audio files in a directory.

    Args:
        files_path: Directory containing MP3 files
        target: Target class label (0, 1, or 2)
        genre_name: Name of the genre for labeling
        smile: OpenSMILE Smile object

    Returns:
        DataFrame containing features from all files
    """
    audio_files = sorted(glob(os.path.join(files_path, '*.mp3')))
    print(f'Extracting features from {len(audio_files)} {genre_name} files')

    dfs = []

    for i, audio_file in enumerate(audio_files):
        try:
            group = f"{genre_name}_{i}"
            cur_df = extract_features(audio_file, smile, group, target)
            dfs.append(cur_df)

            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files")

        except Exception as e:
            print(f"  Error processing {audio_file}: {e}")
            continue

    return pd.concat(dfs)


def main():
    """Command-line interface for feature extraction."""
    parser = argparse.ArgumentParser(
        description='Extract acoustic features from music files'
    )
    parser.add_argument(
        'input_dirs',
        nargs=3,
        help='Three directories containing MP3 files for each genre (salsa soca reggae)'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save extracted feature CSV files'
    )
    parser.add_argument(
        '--genre-names',
        nargs=3,
        default=['salsa', 'soca', 'reggae'],
        help='Names for the three genres (default: salsa soca reggae)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize OpenSMILE
    print("Initializing OpenSMILE with eGeMAPSv02 feature set...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # Extract features for each genre
    for i, (input_dir, genre_name) in enumerate(zip(args.input_dirs, args.genre_names)):
        print(f"\n{'='*60}")
        print(f"Processing {genre_name} (class {i})")
        print(f"{'='*60}")

        df = extract_concatenate_per_class(input_dir, i, genre_name, smile)

        # Save to CSV
        output_file = os.path.join(args.output_dir, f'{genre_name}_concatenated.csv')
        df.to_csv(output_file)
        print(f"Saved features to {output_file}")
        print(f"Shape: {df.shape}")

    print("\nFeature extraction complete!")


if __name__ == '__main__':
    main()
