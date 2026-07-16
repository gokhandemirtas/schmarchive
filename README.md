# Schmarchive — Photo Archive Tool

A CLI tool to **deduplicate**, **geotag**, **organize**, and **normalize** your photo library.

## Requirements

- Python 3.14+
- An OpenAI-compatible API server running locally with a vision model (e.g. LM Studio, Ollama, vLLM, etc.)
- Internet connection (for geocoding via OpenStreetMap Nominatim)

Install dependencies:

```bash
uv sync
```

## Usage

```bash
uv run schmarchive.py
```

You'll be prompted to select a photo folder, then choose from:

```
1 — Geotag files (rename by location/date)
2 — Deduplication (find duplicates via hash + LLM)
3 — Move duplicates to ./duplicates
4 — Normalize filenames (replace non-ASCII chars)
5 — Organize photos into subfolders
q — Quit
```

---

## Features

### 1. Geotag (rename by location/date)

Renames photos based on their EXIF GPS data and date taken.

**How it works:**

1. **Phase 1** — Scans all images, extracts GPS coordinates from EXIF metadata, and groups nearby photos using haversine distance (default radius: 1 km). No API calls are made in this phase.

2. **Phase 2** — Tests the Nominatim geocoder API to ensure it's reachable before proceeding.

3. **Phase 3** — Resolves location names for each unique location group via the Nominatim reverse geocoding API. Each group centroid gets one API call, with rate limiting (1.1s delay between calls). Resolved locations are saved to `locations.csv` for future runs.

4. **Phase 4** — Renames all files using the resolved location names and EXIF dates. No API calls are made in this phase.

**Naming format:**
- Single photo at location on a day: `location-2025-03-24.jpg`
- Multiple photos at same location on same day: `location-2025-03-24-1.jpg`, `location-2025-03-24-2.jpg`
- Photos without GPS: `2025-03-24-14-30.jpg` (date/time only)

### 2. Deduplication (find duplicates)

Finds duplicate and near-duplicate photos using hash comparison + LLM visual analysis.

**Two-pass detection:**

1. **Exact duplicates** — Files with identical MD5 hashes (byte-for-byte copies).
2. **Near-duplicates** — Files with similar perceptual hashes (pHash), indicating visually similar content.

**LLM review:**

Each group of potential duplicates is sent to the LLM server's vision model. The LLM:
- Confirms whether images are truly duplicates
- Identifies the best quality image (sharpest, best exposure, best composition)
- Provides a brief reason for its decision

**Renaming convention:**
- Exact duplicates: `photo__exact_dupe.jpg`
- Near-duplicates: `photo__near_dupe.jpg`

The "best" image keeps its original name.

### 3. Move duplicates

Moves all renamed duplicates (`__exact_dupe`, `__near_dupe`) to a `./duplicates` folder, preserving the directory structure. Traverses all subdirectories recursively.

### 4. Normalize filenames

Replaces non-ASCII characters in filenames with their ASCII equivalents. Scans all images recursively, shows a preview of changes, and renames on confirmation.

Examples:
- `Córdoba-2025-07-03.jpg` → `Cordoba-2025-07-03.jpg`
- `München-2025-01-01.jpg` → `Munchen-2025-01-01.jpg`
- `Śląsk-2025-06-15.jpg` → `Slask-2025-06-15.jpg`

Covers the entire Latin Extended Unicode block (0x00C0–0x024F) plus Turkish characters.

### 5. Organize photos into subfolders

Organizes photos into a `Year/Month/Location` directory structure.

**Submenu options:**

1. **Year / Month / Location** — Parses filenames to extract date and location, then moves files into subfolders.

**Naming patterns recognized:**
- `location-YYYY-MM-DD.jpg` → `2025/March/cordoba/`
- `location-YYYY-MM-DD-N.jpg` → `2025/March/cordoba/`
- `YYYY-MM-DD-HH-MM.jpg` → `2025/March/`
- `IMG_YYYYMMDD_HHMMSS.jpg` → `2025/March/`
- `VID_YYYYMMDD_HHMMSS.jpg` → `2025/March/`
- `PXL_YYYYMMDD_HHMMSS.jpg` → `2025/March/`
- `MVIMGYYYYMMDD_HHMMSS.jpg` → `2025/March/`
- `PANO_YYYYMMDD_HHMMSS.jpg` → `2025/March/`

Files already in a `YYYY/Month/` structure are skipped. All directories are created before moving files. Files that don't match any pattern are reported as unparsed.

Months use named format: January, February, March, etc.

---

## Configuration

### config.json

Create a `config.json` in the project root to override defaults:

```json
{
  "llm_url": "http://localhost:1234/v1/chat/completions",
  "llm_model": "qwen/qwen2.5-vl-7b",
  "llm_timeout": 60,
  "geocode_radius_km": 1.0,
  "geocode_delay_sec": 1.1,
  "geocode_retries": 3,
  "near_duplicate_threshold": 8,
  "blur_threshold": 50.0
}
```

### Environment variables

All settings can be overridden with `SCHMARCHIVE_` prefixed env vars:

| Env Var | Default | Description |
|---------|---------|-------------|
| `SCHMARCHIVE_LLM_URL` | `http://localhost:1234/v1/chat/completions` | LLM API endpoint (OpenAI-compatible) |
| `SCHMARCHIVE_LLM_MODEL` | `qwen/qwen2.5-vl-7b` | Vision model to use |
| `SCHMARCHIVE_LLM_TIMEOUT` | `60` | LLM request timeout (seconds) |
| `SCHMARCHIVE_GEOCODE_RADIUS_KM` | `1.0` | Proximity radius to group nearby photos (km) |
| `SCHMARCHIVE_GEOCODE_DELAY_SEC` | `1.1` | Delay between geocoding API calls (seconds) |
| `SCHMARCHIVE_GEOCODER_TIMEOUT` | `10` | Geocoder API timeout (seconds) |
| `SCHMARCHIVE_NEAR_DUPLICATE_THRESHOLD` | `8` | pHash distance threshold for near-duplicates |
| `SCHMARCHIVE_BLUR_THRESHOLD` | `50.0` | Laplacian variance threshold for blur detection |

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

### Filename Normalization

Uses Unicode NFD decomposition to strip accents from the entire Latin Extended block (0x00C0–0x024F), plus explicit handling for Turkish characters (dotless-i, etc.) that NFD cannot decompose.

### Path Sandboxing

All file operations are sandboxed to the selected folder. The tool will refuse to rename or move files outside the sandbox, preventing accidental data loss.

---

## Project Structure

```
dedupe/
├── schmarchive.py     # Main script
├── pyproject.toml     # Project metadata and dependencies
├── config.json        # User configuration (optional, not committed)
├── locations.csv      # Cached geocode results (auto-generated)
├── photos/            # Your photo library (input)
│   ├── 2025/
│   │   ├── March/
│   │   │   └── cordoba/
│   │   ├── June/
│   │   │   └── istanbul/
│   │   └── July/
│   │       └── cadiz/
│   └── ...
├── duplicates/        # Moved duplicate files (output)
└── videos/            # Video files (not currently processed)
```

---

## Notes

- The tool processes images (jpg, jpeg, png, webp). Videos are not currently supported.
- Files already marked as duplicates (`__exact_dupe`, `__near_dupe`) are skipped during scanning.
- The geocoder uses OpenStreetMap's Nominatim API. Heavy use may result in temporary rate limiting.
- The LLM requires a local OpenAI-compatible API server with a vision model loaded.
- The organize feature only moves files from the root of the selected folder (not from existing subfolders).
