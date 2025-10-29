# Data Directory

This directory contains the query lists used for downloading music from YouTube and the extracted acoustic features for all three genres.

## Structure

```
data/
├── queries/
│   ├── salsa_queries.csv
│   ├── soca_queries.csv
│   └── reggae_queries.csv
├── salsa_concatenated.csv
├── soca_concatenated.csv
└── reggae_concatenated.csv
```

## Query Files

The query CSV files contain lists of artists for each genre:
- `salsa_queries.csv`: 63 salsa artists
- `soca_queries.csv`: 100 soca artists
- `reggae_queries.csv`: 100 reggae artists

Each file has a single column "Youtube Prompts" containing artist names used to search and download songs from YouTube.

## Feature Files

The concatenated feature CSV files are included in this repository. Each file contains:

**File Information:**
- `salsa_concatenated.csv`: 2,992 samples from 748 songs (2.6 MB)
- `soca_concatenated.csv`: 4,956 samples from 1,239 songs (4.3 MB)
- `reggae_concatenated.csv`: 3,264 samples from 816 songs (2.9 MB)

**Feature Content:**
- 87 acoustic features (eGeMAPSv02 feature set from OpenSMILE)
- Metadata columns: `group` (song identifier), `token` (segment identifier), `target` (class label)
- Multiple rows per song (4 samples per song, each representing a 10-second segment)

**Feature Set Details:**
The eGeMAPSv02 feature set includes functionals (statistical summaries) of low-level acoustic descriptors such as:
- Frequency-related parameters (pitch, formants, spectral characteristics)
- Energy/amplitude-related parameters (loudness, dynamics)
- Spectral balance parameters
- Temporal features (voice quality, rhythm)

These features were extracted from 10-second audio segments using OpenSMILE's standard configuration.

Note: The original MP3 audio files are not included in version control due to their large size and copyright considerations (see `.gitignore`).
