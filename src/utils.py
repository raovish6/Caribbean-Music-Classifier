"""
Utility functions for music genre classification project.

This module contains shared helper functions used across different scripts.
"""

import os
import numpy as np
import pandas as pd


def validate_audio_directory(directory):
    """
    Validate that a directory exists and contains audio files.

    Args:
        directory: Path to directory to validate

    Returns:
        bool: True if directory is valid, False otherwise
    """
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return False

    mp3_files = [f for f in os.listdir(directory) if f.endswith('.mp3')]
    if len(mp3_files) == 0:
        print(f"Warning: No MP3 files found in {directory}")
        return False

    return True


def validate_csv_file(filepath):
    """
    Validate that a CSV file exists and can be read.

    Args:
        filepath: Path to CSV file

    Returns:
        bool: True if file is valid, False otherwise
    """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        return False

    try:
        df = pd.read_csv(filepath)
        return True
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False


def print_data_summary(X, y, groups):
    """
    Print a summary of loaded data.

    Args:
        X: Feature matrix
        y: Target labels
        groups: Group identifiers
    """
    print("\nData Summary:")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Number of features: {X.shape[1]}")
    print(f"  Number of unique groups: {len(np.unique(groups))}")

    unique, counts = np.unique(y, return_counts=True)
    print(f"  Class distribution:")
    for label, count in zip(unique, counts):
        print(f"    Class {int(label)}: {count} samples ({count/len(y)*100:.1f}%)")


def ensure_directory(directory):
    """
    Create directory if it doesn't exist.

    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)
