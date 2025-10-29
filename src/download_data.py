"""
Download audio data from YouTube based on artist query lists.

This module downloads songs from YouTube using yt-dlp, filtering by duration
and converting to MP3 format. It's designed to build a dataset for music
genre classification.
"""

import os
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yt_dlp as youtube_dl
from pytube import Search


def longer_than_five_minutes(info, *, incomplete):
    """
    Filter function for yt-dlp to download only videos of appropriate length.

    Args:
        info: Video information dictionary from yt-dlp
        incomplete: Flag indicating if download is incomplete

    Returns:
        Error message string if video should be skipped, None otherwise
    """
    duration = info.get('duration')
    if duration and duration > 300:
        return 'The video is too long'
    if duration and duration < 60:
        return 'The video is too short'
    return None


def download_audio(yt_url, direc):
    """
    Download a single YouTube video as MP3 audio.

    Args:
        yt_url: YouTube URL to download
        direc: Directory to save the downloaded audio
    """
    ydl_opts = {
        'format': 'bestaudio/best[filesize<2M]',
        'outtmpl': os.path.join(direc, '%(title)s.%(ext)s'),
        'match_filter': longer_than_five_minutes,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([yt_url])
        except Exception as e:
            print(f"Failed to download {yt_url}: {e}")


def download_youtube(query, num_vids, path, author=None):
    """
    Download multiple videos from YouTube based on a search query.

    Args:
        query: Search query string
        num_vids: Number of videos to attempt to download
        path: Directory path to save downloads
        author: Optional artist name to filter results
    """
    s = Search(query)

    # Get list of already downloaded files
    present_titles = []
    if os.path.exists(path):
        present_titles = [
            os.path.splitext(f)[0]
            for f in os.listdir(path)
            if f.endswith('.mp3')
        ]

    running_titles = []

    # Fetch enough search results
    while len(s.results) < num_vids:
        s.get_next_results()

    # Download videos with filtering
    for i in range(num_vids):
        if i >= len(s.results):
            break

        result = s.results[i]

        # Filter by author if specified
        if author is not None and result.author != author:
            continue

        url = result.watch_url

        # Clean the title for filename
        title = "".join(
            x for x in result.title
            if (x.isalnum() or x in "._- ")
        )

        # Skip if already downloaded
        if title in present_titles or title in running_titles:
            continue

        running_titles.append(title)

        try:
            download_audio(url, path)
            print(f"Downloaded: {title}")
        except Exception as e:
            print(f"Skipped: {title} ({e})")
            continue

        time.sleep(0.1)


def run_queries(save_path, queries_path):
    """
    Download songs for all artists listed in a CSV file.

    Args:
        save_path: Directory to save downloaded songs
        queries_path: Path to CSV file containing artist queries
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Read artist queries
    query_df = pd.read_csv(queries_path)
    artists = query_df['Youtube Prompts'].dropna().to_list()

    print(f"Downloading songs for {len(artists)} artists to {save_path}")

    for i, artist in enumerate(artists):
        print(f"\n[{i+1}/{len(artists)}] Processing: {artist}")
        download_youtube(artist, 20, save_path, None)


def main():
    """Command-line interface for downloading music data."""
    parser = argparse.ArgumentParser(
        description='Download music from YouTube for genre classification'
    )
    parser.add_argument(
        'queries_csv',
        type=str,
        help='Path to CSV file containing artist queries'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save downloaded MP3 files'
    )
    parser.add_argument(
        '--num-vids',
        type=int,
        default=20,
        help='Number of videos to download per artist (default: 20)'
    )

    args = parser.parse_args()

    run_queries(args.output_dir, args.queries_csv)

    print("\nDownload complete!")


if __name__ == '__main__':
    main()
