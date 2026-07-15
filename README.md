# Dedupe — Photo Archive Tool

A CLI tool to **deduplicate**, **geotag**, and **organize** your photo library.

## Requirements

- Python 3.14+
- [LM Studio](https://lmstudio.ai) running locally with a vision model (default: `qwen/qwen2.5-vl-7b`)
- Internet connection (for geocoding via OpenStreetMap Nominatim)

Install dependencies:

```bash
uv sync
```

## Usage

```bash
uv run dedupe.py
```

You'll be prompted to select a photo folder, then choose from:

```
1 — Geotag files (rename by location/date)
2 — Dedupe (find duplicates via hash + LLM)
3 — Move deduped files to ./duplicates
q — Quit
```

---

## Features

### 1. Geotag (rename by location/date)

Renames photos based on their EXIF GPS data and date taken.

**How it works:**

1. **Phase 1** — Scans all images, extracts GPS coordinates from EXIF metadata, and groups nearby photos using haversine distance (default radius: 1 km). No API calls are made in this phase.

2. **Phase 2** — Tests the Nominatim geocoder API to ensure it's reachable before proceeding.

3. **Phase 3** — Resolves location names for each unique location group via the Nominatim reverse geocoding API. Each group centroid gets one API call, with rate limiting (1.1s delay between calls).

4. **Phase 4** — Renames all files using the resolved location names and EXIF dates. No API calls are made in this phase.

**Naming format:**
- Single photo at location on a day: `location-2025-03-24.jpg`
- Multiple photos at same location on same day: `location-2025-03-24-1.jpg`, `location-2025-03-24-2.jpg`
- Photos without GPS: `2025-03-24-14-30.jpg` (date/time only)

**Turkish characters** are transliterated to English (ş→s, ç→c, ğ→g, ı→i, ö→o, ü→u).

### 2. Dedupe (find duplicates)

Finds duplicate and near-duplicate photos using hash comparison + LLM visual analysis.

**Two-pass detection:**

1. **Exact duplicates** — Files with identical MD5 hashes (byte-for-byte copies).
2. **Near-duplicates** — Files with similar perceptual hashes (pHash), indicating visually similar content.

**LLM review:**

Each group of potential duplicates is sent to LM Studio's vision model. The LLM:
- Confirms whether images are truly duplicates
- Identifies the best quality image (sharpest, best exposure, best composition)
- Provides a brief reason for its decision

**Renaming convention:**
- Exact duplicates: `photo__exact_dupe.jpg`
- Near-duplicates: `photo__near_dupe.jpg`

The "best" image keeps its original name.

### 3. Move deduped files

Moves all renamed duplicates (`__exact_dupe`, `__near_dupe`) to a `./duplicates` folder, preserving the directory structure.

---

## Configuration

### config.json

Create a `config.json` in the project root to override defaults:

```json
{
  "lm_studio_url": "http://localhost:1234/v1/chat/completions",
  "lm_studio_model": "qwen/qwen2.5-vl-7b",
  "lm_studio_timeout": 60,
  "geocode_radius_km": 1.0,
  "geocode_delay_sec": 1.1,
  "geocode_retries": 3,
  "near_duplicate_threshold": 8,
  "blur_threshold": 50.0
}
```

### Environment variables

All settings can be overridden with `DEDUPE_` prefixed env vars:

| Env Var | Default | Description |
|---------|---------|-------------|
| `DEDUPE_LM_STUDIO_URL` | `http://localhost:1234/v1/chat/completions` | LM Studio API endpoint |
| `DEDUPE_LM_STUDIO_MODEL` | `qwen/qwen2.5-vl-7b` | Vision model to use |
| `DEDUPE_LM_STUDIO_TIMEOUT` | `60` | LLM request timeout (seconds) |
| `DEDUPE_GEOCODE_RADIUS_KM` | `1.0` | Proximity radius to group nearby photos (km) |
| `DEDUPE_GEOCODE_DELAY_SEC` | `1.1` | Delay between geocoding API calls (seconds) |
| `DEDUPE_GEOCODER_TIMEOUT` | `10` | Geocoder API timeout (seconds) |
| `DEDUPE_NEAR_DUPLICATE_THRESHOLD` | `8` | pHash distance threshold for near-duplicates |
| `DEDUPE_BLUR_THRESHOLD` | `50.0` | Laplacian variance threshold for blur detection |

**Priority:** env vars > config.json > built-in defaults

---

## How It Works

### Geotagging — Proximity Caching

Instead of calling the geocoding API for every photo, the tool:

1. Groups photos by GPS proximity using haversine distance
2. Resolves one API call per unique location group
3. Reuses the location name for all photos in that group

This reduces API calls from N (one per photo) to U (one per unique location), respecting Nominatim's 1 req/sec rate limit.

**CSV persistence:** Resolved locations are saved to `locations.csv`. On subsequent runs, known coordinates are looked up from the CSV before hitting the API — making repeat geotagging essentially free.

### Deduplication — Hash + LLM Pipeline

1. **Scan** — Compute MD5 hash and perceptual hash (pHash) for each image
2. **Cluster** — Group by exact hash match, then by pHash distance
3. **Verify** — Send each group to the LLM for visual confirmation
4. **Mark** — Rename non-best images with dupe suffixes

### Path Sandboxing

All file operations are sandboxed to the selected folder. The tool will refuse to rename or move files outside the sandbox, preventing accidental data loss.

---

## Project Structure

```
dedupe/
├── dedupe.py          # Main script
├── pyproject.toml     # Project metadata and dependencies
├── config.json        # User configuration (optional, not committed)
├── locations.csv      # Cached geocode results (auto-generated)
├── photos/            # Your photo library (input)
├── duplicates/        # Moved duplicate files (output)
└── videos/            # Video files (not currently processed)
```

---

## Notes

- The tool only processes images (jpg, jpeg, png, webp). Videos are not supported.
- Files already marked as duplicates (`__exact_dupe`, `__near_dupe`) are skipped during scanning.
- The geocoder uses OpenStreetMap's Nominatim API. Heavy use may result in temporary rate limiting.
- The LLM requires a local LM Studio instance with a vision model loaded.
