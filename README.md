# Schmarchive — Photo Archive Tool

```
    ____       _                              _     _           
   / ___|  ___| |__  _ __ ___   __ _ _ __ ___| |__ (_)_   _____ 
   \___ \ / __| '_ \| '_ ` _ \ / _` | '__/ __| '_ \ \ \ / / _ \
    ___) | (__| | | | | | | | | (_| | | | (__| | | | |\ V /  __/
   |____/ \___|_| |_|_| |_| |_|\__,_|_|  \___|_| |_|_| \_/ \___|
```

A CLI tool to **deduplicate**, **geotag**, **organize**, **identify**, and **tame** your photo library.

## Requirements

- Python 3.14+
- An OpenAI-compatible inference server running locally (e.g. LM Studio, Ollama, vLLM, etc.) / or cloud with a vision model 
- Internet connection (for geocoding via OpenStreetMap Nominatim). In order to avoid hammering Nominatim, photos are clustered based on proximity. If you abuse it, your they'll ban you.

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
2 — Date-tag files (rename by date only)
3 — Deduplication (find duplicates via hash + LLM)
4 — Move duplicates
5 — Normalize filenames (replace non-ASCII chars)
6 — Organize photos into subfolders
7 — Identify landmarks (rename by building/tourism)
8 — Pluck images by subject (LLM-powered)
9 — Configure settings
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
- Single photo at location on a day: `cordoba-2025-03-24.jpg`
- Multiple photos at same location on same day: `cordoba-2025-03-24-1.jpg`, `cordoba-2025-03-24-2.jpg`
- Photos without GPS: `2025-03-24-14-30.jpg` (date/time only)

### 2. Date-tag (rename by date only)

Renames photos using their EXIF date taken (or file creation date as fallback). No API calls needed — runs instantly.

**Naming format:** `YYYY-MM-DD-HH-MM.ext`

When multiple photos share the same minute, a collision suffix is added: `2025-07-03-14-30-1.jpg`, `2025-07-03-14-30-2.jpg`, etc.

### 3. Deduplication (find duplicates)

Finds duplicate and near-duplicate photos using a two-stage pipeline: fast local hashing followed by LLM visual verification.

**Stage 1 — Cluster by hash (local, fast):**

1. **Exact duplicates** — Files with identical MD5 hashes (byte-for-byte copies) are grouped together.
2. **Near-duplicates** — Files are assigned a perceptual hash (pHash), a 64-bit fingerprint of visual content. Images whose pHashes differ by 8 or fewer bits are grouped as near-duplicate candidates. The `near_duplicate_threshold` setting controls this sensitivity.

This stage is a fast pre-filter — it produces candidate groups without any API calls.

**Stage 2 — LLM verification (visual analysis):**

Each candidate group is sent to the LLM's vision model for confirmation. The LLM:
- Confirms whether images are truly duplicates (pHash can produce false positives)
- Identifies the best quality image (sharpest, best exposure, best composition)
- Provides a brief reason for its decision

Only groups confirmed by the LLM are marked. This prevents false positives from the pHash pre-filter.

**Renaming convention:**
- Exact duplicates: `photo__exact_dupe.jpg`
- Near-duplicates: `photo__near_dupe.jpg`

The "best" image keeps its original name.

### 4. Move duplicates

Moves all renamed duplicates (`__exact_dupe`, `__near_dupe`) to a `./duplicates` folder, preserving the directory structure. Traverses all subdirectories recursively.

### 5. Normalize filenames

Replaces non-ASCII characters in filenames with their ASCII equivalents. Scans all images recursively, shows a preview of changes, and renames on confirmation.

Examples:
- `Córdoba-2025-07-03.jpg` → `Cordoba-2025-07-03.jpg`
- `München-2025-01-01.jpg` → `Munchen-2025-01-01.jpg`
- `Śląsk-2025-06-15.jpg` → `Slask-2025-06-15.jpg`

Covers the entire Latin Extended Unicode block (0x00C0–0x024F) plus Turkish characters.

### 6. Organize photos into subfolders

Organizes photos into subfolder structures. Contains a submenu:

```
1 — Year / Month / Location
2 — AI subject categorization
q — Back
```

#### 6a. Year / Month / Location

Parses filenames to extract date and location, then moves files into a `Year/Month/Country/City/` directory structure.

**Naming patterns recognized:**
- `country-city-YYYY-MM-DD.jpg` → `2025/July/spain/cordoba/`
- `city-YYYY-MM-DD.jpg` → `2025/March/cordoba/`
- `YYYY-MM-DD-HH-MM.jpg` → `2025/March/`
- `IMG_YYYYMMDD_HHMMSS.jpg` → `2025/March/`
- `VID_YYYYMMDD_HHMMSS.jpg` → `2025/March/`
- `PXL_YYYYMMDD_HHMMSS.jpg` → `2025/March/`
- `MVIMGYYYYMMDD_HHMMSS.jpg` → `2025/March/`
- `PANO_YYYYMMDD_HHMMSS.jpg` → `2025/March/`

Files already in a `YYYY/Month/` structure are skipped. All directories are created before moving files. Months use named format: January, February, March, etc.

#### 6b. AI subject categorization

Uses the LLM to categorize each photo into a subject category, then organizes them into category subfolders.

**Default categories:** animals, architecture, cityscape, food, landscape, night, people, portrait, selfie, street, travel, vehicle

Categories are configurable via `photo_categories` in Config.

### 7. Identify landmarks (rename by building/tourism)

Queries the Nominatim API with each photo's GPS coordinates to find nearby landmarks. Renames files using the most specific location name available in this priority order:

1. `building` (e.g., `eiffel-tower.jpg`)
2. `tourism` (e.g., `grand-canyon.jpg`)
3. `historic` (e.g., `stonehenge.jpg`)
4. `natural` (e.g., `mount-everest.jpg`)

Photos without a recognized landmark are left unchanged. Rate-limited with configurable delay between API calls.

### 8. Pluck images by subject (LLM-powered)

Extracts photos matching a user-provided subject description using the LLM.

**How it works:**

1. Enter a subject (max 50 characters, ASCII only), e.g. `"white cat"`
2. The LLM scans each root-level image asking: *"Does this image contain a white cat?"*
3. Matching images are previewed, then moved to a `{subject}/` folder (e.g., `white-cat/`)

---

## Configuration

### config.json

Create a `config.json` in the project root to override defaults:

```json
{
  "llm_url": "http://localhost:1234/v1/chat/completions",
  "llm_model": "qwen/qwen2.5-vl-7b",
  "llm_timeout": 60,
  "llm_api_key": "",
  "geocode_radius_km": 1.0,
  "geocode_delay_sec": 1.1,
  "geocode_retries": 3,
  "near_duplicate_threshold": 8,
  "blur_threshold": 50.0,
  "photo_categories": ["animals", "architecture", "cityscape", "food", "landscape", "night", "people", "portrait", "selfie", "street", "travel", "vehicle"],
  "log_file": "schmarchive.log"
}
```

### Environment variables

All settings can be overridden with `SCHMARCHIVE_` prefixed env vars:

| Env Var | Default | Description |
|---------|---------|-------------|
| `SCHMARCHIVE_LLM_URL` | `http://localhost:1234/v1/chat/completions` | LLM API endpoint (OpenAI-compatible) |
| `SCHMARCHIVE_LLM_MODEL` | `qwen/qwen2.5-vl-7b` | Vision model to use |
| `SCHMARCHIVE_LLM_TIMEOUT` | `60` | LLM request timeout (seconds) |
| `SCHMARCHIVE_LLM_API_KEY` | `""` | API key for cloud providers (sent as Bearer token) |
| `SCHMARCHIVE_GEOCODE_RADIUS_KM` | `1.0` | Proximity radius to group nearby photos (km) |
| `SCHMARCHIVE_GEOCODE_DELAY_SEC` | `1.1` | Delay between geocoding API calls (seconds) |
| `SCHMARCHIVE_GEOCODER_TIMEOUT` | `10` | Geocoder API timeout (seconds) |
| `SCHMARCHIVE_NEAR_DUPLICATE_THRESHOLD` | `8` | pHash distance threshold for near-duplicates |
| `SCHMARCHIVE_BLUR_THRESHOLD` | `50.0` | Laplacian variance threshold for blur detection |
| `SCHMARCHIVE_LOG_FILE` | `schmarchive.log` | Path to the log file |

**Priority:** env vars > config.json > built-in defaults

---

## Logging

All operations are logged to `schmarchive.log` (configurable via `log_file`). Each entry includes a timestamp:

```
[2025-07-03 14:30:00] schmarchive started
[2025-07-03 14:30:01] Geotag started: Z:\photos
[2025-07-03 14:30:05] Renamed: DSCN4800.JPG → spain-cordoba-2025-07-03.jpg
[2025-07-03 14:31:00] Geotag done: 200 GPS, 26 date-fallback
[2025-07-03 14:31:05] schmarchive quit
```

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
├── schmarchive.log    # Operation log (auto-generated)
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
- The organize and pluck features only move files from the root of the selected folder (not from existing subfolders).

### LLM API Compatibility

The tool expects **OpenAI-compatible** API responses. Your endpoint must return the standard `/v1/chat/completions` response structure:

```json
{
  "choices": [
    {
      "message": {
        "content": "{ \"key\": \"value\" }"
      }
    }
  ]
}
```

The tool parses `choices[0].message.content` as the LLM's response. This works with:
- **Local servers:** LM Studio, Ollama, vLLM, llama.cpp server, LocalAI
- **Cloud providers:** OpenAI, Together AI, Groq, Fireworks, Azure OpenAI, and any provider offering an OpenAI-compatible API

Set `llm_api_key` (or `SCHMARCHIVE_LLM_API_KEY`) for cloud providers that require authentication. The key is sent as a `Bearer` token in the `Authorization` header.

If your provider uses a different response format, the tool will fail with an LLM error.
