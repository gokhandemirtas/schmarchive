import os
import hashlib
import shutil
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import imagehash
import pandas as pd
import requests
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS, IFD
from geopy.geocoders import Nominatim
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

console = Console()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FILE = "schmarchive.log"
_LOG_MAX_LINES = 2000


def log(msg: str):
    """Append a timestamped line to the log file, with auto-trim."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
        # Auto-trim if log exceeds max lines
        if _LOG_MAX_LINES > 0:
            with open(_LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > _LOG_MAX_LINES:
                with open(_LOG_FILE, "w", encoding="utf-8") as f:
                    f.writelines(lines[-_LOG_MAX_LINES:])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    llm_url: str = "http://localhost:1234/v1/chat/completions"
    llm_model: str = "qwen/qwen2.5-vl-7b"
    llm_timeout: int = 60
    llm_api_key: str = ""

    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")

    geocoder_user_agent: str = "schmarchive_photo_tool"
    geocoder_timeout: int = 10
    geocode_radius_km: float = 1.0
    geocode_delay_sec: float = 1.1
    geocode_retries: int = 3

    near_duplicate_threshold: int = 8
    blur_threshold: float = 50.0

    photo_categories: tuple = (
        "animals",
        "architecture",
        "cityscape",
        "food",
        "landscape",
        "night",
        "people",
        "portrait",
        "selfie",
        "street",
        "travel",
        "vehicle",
    )

    log_file: str = "schmarchive.log"
    log_max_lines: int = 2000

    photo_folder: str = "./photos"
    duplicates_folder: str = "duplicates"

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


config = Config()


def _llm_headers():
    """Build request headers with optional Bearer token for cloud providers."""
    headers = {"Content-Type": "application/json"}
    if config.llm_api_key:
        headers["Authorization"] = f"Bearer {config.llm_api_key}"
    return headers


# ---------------------------------------------------------------------------
# Location cache (proximity-based geocode dedup)
# ---------------------------------------------------------------------------

class LocationCache:
    """Stores resolved (lat, lon, name) tuples and reuses names for nearby coords."""

    def __init__(self, radius_km: float = 1.0, csv_path: Optional[str] = None):
        self.radius_km = radius_km
        self.csv_path = csv_path or str(Path(__file__).parent / "locations.csv")
        self._entries: list[tuple[float, float, str]] = []
        self._load_csv()

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def get(self, lat: float, lon: float) -> tuple[Optional[str], Optional[str]]:
        """Returns (name, country) if a nearby entry exists, else (None, None)."""
        for entry in self._entries:
            elat, elon = entry[0], entry[1]
            if self._haversine_km(lat, lon, elat, elon) <= self.radius_km:
                return entry[2], entry[3] if len(entry) > 3 else ""
        return None, None

    def add(self, lat: float, lon: float, name: str, country: str = ""):
        self._entries.append((lat, lon, name, country))

    def _load_csv(self):
        """Load previously resolved locations from CSV."""
        try:
            with open(self.csv_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("lat"):
                        continue
                    parts = line.split(",", 3)
                    if len(parts) >= 3:
                        try:
                            lat, lon = float(parts[0]), float(parts[1])
                            name = parts[2]
                            country = parts[3] if len(parts) > 3 else ""
                            self._entries.append((lat, lon, name, country))
                        except ValueError:
                            continue
        except FileNotFoundError:
            pass

    def save_csv(self):
        """Persist all resolved locations to CSV."""
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("lat,lon,name,country\n")
            for lat, lon, name, country in self._entries:
                f.write(f"{lat:.6f},{lon:.6f},{name},{country}\n")

    def group_by_proximity(self, coords: list[tuple[float, float]]) -> list[list[int]]:
        """Group indices of coords that are within radius_km of each other.
        Returns list of groups, each group is a list of indices into coords."""
        n = len(coords)
        visited = set()
        groups = []
        for i in range(n):
            if i in visited:
                continue
            group = [i]
            for j in range(i + 1, n):
                if j in visited:
                    continue
                if self._haversine_km(coords[i][0], coords[i][1],
                                       coords[j][0], coords[j][1]) <= self.radius_km:
                    group.append(j)
            if len(group) > 1:
                groups.append(group)
                visited.update(group)
        return groups


# ---------------------------------------------------------------------------
# Globals using config
# ---------------------------------------------------------------------------

_geocoder: Optional[Nominatim] = None
_location_cache_obj: Optional[LocationCache] = None


def _get_geocoder() -> Nominatim:
    global _geocoder
    if _geocoder is None:
        _geocoder = Nominatim(user_agent=config.geocoder_user_agent, timeout=config.geocoder_timeout)
    return _geocoder


def _get_location_cache() -> LocationCache:
    global _location_cache_obj
    if _location_cache_obj is None:
        _location_cache_obj = LocationCache(radius_km=config.geocode_radius_km)
    return _location_cache_obj


# ---------------------------------------------------------------------------
# Path sandboxing
# ---------------------------------------------------------------------------

class PathEscapeError(Exception):
    """Raised when an operation would escape the sandbox."""


def sandbox_resolve(folder, target_path):
    """Resolve target_path and verify it lives inside folder.
    Returns the resolved absolute path or raises PathEscapeError."""
    folder_resolved = Path(folder).resolve()
    target_resolved = (folder_resolved / target_path).resolve() if not os.path.isabs(target_path) else Path(target_path).resolve()
    try:
        target_resolved.relative_to(folder_resolved)
    except ValueError:
        raise PathEscapeError(
            f"Path escapes sandbox: {target_path}\n"
            f"  sandbox root: {folder_resolved}\n"
            f"  resolved to:  {target_resolved}"
        )
    return str(target_resolved)


def sandbox_rename(folder, src, dst):
    """Rename src -> dst, both must resolve inside folder."""
    src_resolved = sandbox_resolve(folder, src)
    dst_resolved = sandbox_resolve(folder, dst)
    os.rename(src_resolved, dst_resolved)
    return dst_resolved


def sandbox_move(folder, src, dest_root):
    """Move src (inside folder) to dest_root, preserving relative structure.
    dest_root must be an absolute path. The destination is recomputed and verified."""
    src_resolved = sandbox_resolve(folder, src)
    rel = os.path.relpath(src_resolved, Path(folder).resolve())
    dst = os.path.join(dest_root, rel)
    dst_resolved = str(Path(dst).resolve())
    # Verify destination is inside dest_root
    try:
        Path(dst_resolved).relative_to(str(Path(dest_root).resolve()))
    except ValueError:
        raise PathEscapeError(
            f"Move destination escapes target: {dst_resolved}\n"
            f"  dest root: {dest_root}"
        )
    os.makedirs(os.path.dirname(dst_resolved), exist_ok=True)
    shutil.move(src_resolved, dst_resolved)
    return dst_resolved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask_folder():
    path = prompt("Photo folder [./photos]: ", completer=PathCompleter())
    path = path.strip().strip('"') or "./photos"
    resolved = str(Path(path).resolve())
    if not os.path.isdir(resolved):
        console.print("[red]Folder does not exist.[/red]")
        raise SystemExit(1)
    console.print(f"  Using: [cyan]{resolved}[/cyan]")
    return resolved


import unicodedata


def _build_char_map():
    """Build a comprehensive Latin accent → base letter map."""
    extras = {
        "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
        "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U",
    }
    m = {}
    for cp in range(0x00C0, 0x024F):
        ch = chr(cp)
        decomposed = unicodedata.normalize("NFD", ch)
        base = decomposed[0] if decomposed and decomposed[0].isascii() else ch
        m[ch] = base
    m.update(extras)
    return str.maketrans(m)


_CHAR_MAP = _build_char_map()


def slugify(text):
    text = text.lower().strip().translate(_CHAR_MAP)
    result = []
    for c in text:
        if c.isalnum() or c == " ":
            result.append(c)
    return "".join(result).replace(" ", "")


# ---------------------------------------------------------------------------
# EXIF / GPS
# ---------------------------------------------------------------------------

def _get_exif(path):
    try:
        img = Image.open(path)
        return img._getexif() or {}
    except Exception:
        return {}


def get_gps(path):
    exif = _get_exif(path)
    if not exif:
        return None
    gps_tag = None
    for tag_id, val in exif.items():
        if TAGS.get(tag_id) == "GPSInfo":
            gps_tag = val
            break
    if not gps_tag:
        return None
    gps = {}
    for tag_id, val in gps_tag.items():
        gps[GPSTAGS.get(tag_id, tag_id)] = val

    def to_degrees(val):
        if val is None:
            return None
        d, m, s = val
        result = float(d) + float(m) / 60 + float(s) / 3600
        if not math.isfinite(result):
            return None
        return result

    lat = to_degrees(gps.get("GPSLatitude"))
    lon = to_degrees(gps.get("GPSLongitude"))
    if lat is None or lon is None:
        return None
    if gps.get("GPSLatitudeRef") == "S":
        lat = -lat
    if gps.get("GPSLongitudeRef") == "W":
        lon = -lon
    return (round(lat, 6), round(lon, 6))


def get_date_taken(path):
    exif = _get_exif(path)
    if exif:
        for tag_id in (36867, 36868, 306):  # DateTimeOriginal, DateTimeDigitized, DateTime
            val = exif.get(tag_id)
            if val:
                try:
                    from datetime import datetime
                    return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                except Exception:
                    continue
    from datetime import datetime
    return datetime.fromtimestamp(os.path.getmtime(path))


def reverse_geocode(lat, lon):
    cache = _get_location_cache()
    cached_name, cached_country = cache.get(lat, lon)
    if cached_name is not None:
        if cached_name:
            console.print(f"    [dim](cached: [bold]{cached_name}[/bold], {cached_country})[/dim]")
        return cached_name

    import time

    last_err = None
    for attempt in range(config.geocode_retries):
        try:
            loc = _get_geocoder().reverse((lat, lon), language="en")
            time.sleep(config.geocode_delay_sec)
            if loc and loc.raw.get("address"):
                addr = loc.raw["address"]
                for field in ("city", "town", "village", "hamlet", "suburb", "neighbourhood", "county", "state"):
                    if field in addr:
                        cache.add(lat, lon, addr[field])
                        return addr[field]
                name = loc.address.split(",")[0]
                cache.add(lat, lon, name)
                return name
            cache.add(lat, lon, "")
            return ""
        except Exception as e:
            last_err = e
            wait = config.geocode_delay_sec * (2 ** attempt)
            console.print(f"    [yellow]Geocode error (attempt {attempt + 1}/{config.geocode_retries}): {e}[/yellow]")
            if attempt < config.geocode_retries - 1:
                console.print(f"    [dim]Retrying in {wait:.1f}s...[/dim]")
                time.sleep(wait)

    console.print(f"    [red]Geocode failed after {config.geocode_retries} attempts — skipping this image[/red]")
    cache.add(lat, lon, "")
    return ""


def test_geocoder() -> bool:
    """Quick probe to check if Nominatim is reachable."""
    import time
    try:
        _get_geocoder().reverse((41.0082, 28.9784), language="en")
        time.sleep(config.geocode_delay_sec)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Image scanning
# ---------------------------------------------------------------------------

def collect_images(folder):
    files = [
        os.path.join(root, f)
        for root, _, filenames in os.walk(folder, followlinks=False)
        for f in filenames
        if f.lower().endswith(config.image_extensions)
        and "__exact_dupe" not in f
        and "__near_dupe" not in f
    ]
    records = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning images...", total=len(files))
        for image_path in files:
            progress.update(task, description=f"[cyan]{os.path.basename(image_path)}[/cyan]")
            records.append({
                "path": image_path,
                "folder": os.path.relpath(os.path.dirname(image_path), folder),
                "hash": hashlib.md5(open(image_path, "rb").read()).hexdigest(),
                "phash": calculate_phash(image_path),
                "blurry": is_blurry(image_path),
            })
            progress.advance(task)
    return pd.DataFrame(records)


def calculate_phash(image_path):
    try:
        return imagehash.phash(Image.open(image_path))
    except Exception:
        return None


def is_blurry(image_path, threshold=None):
    if threshold is None:
        threshold = config.blur_threshold
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return bool(cv2.Laplacian(gray, cv2.CV_64F).var() < threshold)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def find_duplicate_groups(df):
    dupes = df[df.duplicated(subset="hash", keep=False)].copy()
    return dupes.groupby("hash")


def find_near_duplicate_groups(df, threshold=None):
    if threshold is None:
        threshold = config.near_duplicate_threshold
    rows = df[df["phash"].notna()].copy()
    if rows.empty:
        return []
    hashes = rows["phash"].tolist()
    paths = rows["path"].tolist()
    visited = set()
    groups = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Clustering by visual similarity...", total=len(hashes))
        for i in range(len(hashes)):
            if i in visited:
                progress.advance(task)
                continue
            group = [i]
            for j in range(i + 1, len(hashes)):
                if j in visited:
                    continue
                if hashes[i] - hashes[j] <= threshold:
                    group.append(j)
            if len(group) > 1:
                groups.append(group)
                visited.update(group)
            progress.advance(task)
    return [[paths[i] for i in g] for g in groups]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def send_to_llm(images, near_duplicate=False):
    import base64, io
    content = []
    for path in images:
        img = Image.open(path)
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    kind = "visually similar (perceptual hash match)" if near_duplicate else "byte-identical (same hash)"
    filenames = ", ".join(os.path.basename(p) for p in images)
    content.append({"type": "text", "text": (
        f"These {len(images)} images are {kind}: {filenames}\n\n"
        "Review them and decide:\n"
        "1. Are they truly duplicates of the same photo?\n"
        "2. Which one is the BEST quality (sharpest, best exposure, best composition)?\n\n"
        "Reply ONLY with valid JSON, no other text:\n"
        '{"is_duplicate": true/false, "best": "filename.jpg", "reason": "brief explanation"}'
    )})

    try:
        r = requests.post(config.llm_url, json={
            "model": config.llm_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": 200,
        }, headers=_llm_headers(), timeout=config.llm_timeout)
        if r.status_code != 200:
            console.print(f"  [red]LLM error ({r.status_code}):[/red] {r.text[:300]}")
            return {"is_duplicate": False, "best": None, "reason": "API error"}
        raw = r.json()["choices"][0]["message"]["content"]
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        import json
        return json.loads(raw)
    except Exception as e:
        console.print(f"  [red]LLM error:[/red] {e}")
        return {"is_duplicate": False, "best": None, "reason": "API unavailable"}


# ---------------------------------------------------------------------------
# Rename / move
# ---------------------------------------------------------------------------

def rename_with_suffix(path, suffix, kept_name=None):
    base, ext = os.path.splitext(path)
    if kept_name:
        kept_base = os.path.splitext(kept_name)[0]
        return f"{os.path.dirname(path)}/{kept_base}-{suffix}{ext}"
    return f"{base}{suffix}{ext}"


def rename_duplicates(paths, suffix, label, kept_name=None, sandbox=None):
    count = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Renaming {label}...", total=len(paths))
        for src in paths:
            dst = rename_with_suffix(src, suffix, kept_name)
            if src != dst and not os.path.exists(dst):
                if sandbox:
                    try:
                        sandbox_rename(sandbox, src, dst)
                        count += 1
                    except PathEscapeError as e:
                        console.print(f"  [red]BLOCKED:[/red] {e}")
                else:
                    os.rename(src, dst)
                    count += 1
            progress.update(task, description=f"[green]{os.path.basename(src)}[/green]")
            progress.advance(task)
    return count


def move_files(paths, dest_root, source_folder):
    moved = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Moving duplicates...", total=len(paths))
        for src in paths:
            try:
                sandbox_move(source_folder, src, dest_root)
                moved += 1
            except PathEscapeError as e:
                console.print(f"  [red]BLOCKED:[/red] {e}")
            progress.update(task, description=f"[green]{os.path.basename(src)}[/green]")
            progress.advance(task)
    return moved


# ---------------------------------------------------------------------------
# Flow: Geotag
# ---------------------------------------------------------------------------

def check_llm():
    models_url = config.llm_url.replace("/chat/completions", "/models")
    try:
        r = requests.get(models_url, timeout=5)
        r.raise_for_status()
        models = [m.get("id", "") for m in r.json().get("data", [])]
        if not models:
            console.print("[red]No models loaded on LLM server.[/red]")
            raise SystemExit(1)
        console.print(f"  LLM server: [green]connected[/green] — model: {models[0]}")
    except requests.ConnectionError:
        console.print(f"[red]Cannot connect to LLM server at {models_url}.[/red]")
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]LLM server check failed — {e}[/red]")
        raise SystemExit(1)


def flow_geotag(folder):
    log(f"Geotag started: {folder}")
    files = [
        os.path.join(root, f)
        for root, _, filenames in os.walk(folder, followlinks=False)
        for f in filenames
        if f.lower().endswith(config.image_extensions)
        and "__exact_dupe" not in f
        and "__near_dupe" not in f
    ]
    if not files:
        console.print("[dim]No images found.[/dim]")
        return

    console.print(f"  Found [yellow]{len(files)}[/yellow] images to geotag.")
    console.print("  [dim]Images with GPS data will be renamed by location.[/dim]")
    console.print("  [dim]Images without GPS will be renamed by date + folder.[/dim]")
    console.print(f"  [dim]Proximity radius: {config.geocode_radius_km} km[/dim]")

    csv_path = str(Path(folder).parent / "locations.csv")
    csv_exists = os.path.exists(csv_path)
    use_existing_csv = False
    if csv_exists:
        console.print(f"\n  [yellow]locations.csv already exists.[/yellow]")
        use_existing_csv = prompt("  Use existing locations? (y/N) ").strip().lower() == "y"
        if use_existing_csv:
            console.print("  [dim]Using cached locations — skipping API calls[/dim]")

    confirm = prompt("\nProceed? [y/N] ").strip().lower()
    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    # ── Phase 1: Extract GPS, group by proximity (no API calls) ──────────
    console.print("\n[bold]Phase 1:[/bold] Extracting GPS metadata...")
    gps_data: list[tuple[str, float, float]] = []  # (path, lat, lon)
    no_gps: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning images...", total=len(files))
        for path in files:
            progress.update(task, description=f"[cyan]{os.path.basename(path)}[/cyan]")
            gps = get_gps(path)
            if gps:
                gps_data.append((path, gps[0], gps[1]))
            else:
                no_gps.append(path)
            progress.advance(task)

    console.print(f"  [dim]{len(gps_data)} with GPS, {len(no_gps)} without GPS[/dim]")

    if not gps_data:
        console.print("[dim]No GPS data found — falling back to date-only renaming.[/dim]")
        _rename_by_date(folder, no_gps)
        return

    # Group nearby GPS coords into location clusters
    cache = _get_location_cache()
    if use_existing_csv:
        cache._entries.clear()
    else:
        cache._entries.clear()  # fresh cache for new run

    coords = [(lat, lon) for _, lat, lon in gps_data]
    proximity_groups = cache.group_by_proximity(coords)

    # Build representative coords for each group (centroid)
    unique_locations: list[tuple[float, float, list[int]]] = []
    grouped_indices = set()
    for group in proximity_groups:
        grouped_indices.update(group)
        avg_lat = sum(coords[i][0] for i in group) / len(group)
        avg_lon = sum(coords[i][1] for i in group) / len(group)
        unique_locations.append((avg_lat, avg_lon, group))

    # Ungrouped images get their own entry
    for i in range(len(coords)):
        if i not in grouped_indices:
            unique_locations.append((coords[i][0], coords[i][1], [i]))

    console.print(f"  [yellow]{len(unique_locations)}[/yellow] unique location(s) to resolve\n")

    resolved_names: dict[int, tuple[str, str]] = {}  # index -> (country_slug, city_slug)

    if use_existing_csv:
        # ── Load from existing CSV ────────────────────────────────────────
        console.print("[bold]Loading locations from CSV...[/bold]")
        cache._load_csv()
        for idx, (lat, lon, indices) in enumerate(unique_locations):
            city_slug, country_slug = cache.get(lat, lon)
            city_slug = city_slug or ""
            country_slug = country_slug or ""
            for i in indices:
                resolved_names[i] = (country_slug, city_slug)
            display_city = city_slug or "[dim]unknown[/dim]"
            display_country = country_slug or "[dim]unknown[/dim]"
            console.print(f"  [green]{display_city}[/green], [cyan]{display_country}[/cyan] — {len(indices)} image(s)")
        console.print()
    else:
        # ── Phase 2: Test API availability ────────────────────────────────
        console.print("[bold]Phase 2:[/bold] Testing geocoder API...")
        if not test_geocoder():
            console.print("[red]Geocoder API is not responding.[/red]")
            console.print("[dim]Try again later or check your network.[/dim]")
            return
        console.print("  [green]API is alive[/green]\n")

        # ── Phase 3: Resolve unique locations ────────────────────────────────
        console.print("[bold]Phase 3:[/bold] Resolving location names...")

        import time
        for idx, (lat, lon, indices) in enumerate(unique_locations):
            loc = _get_geocoder().reverse((lat, lon), language="en")
            time.sleep(config.geocode_delay_sec)

            city = ""
            country = ""
            if loc and loc.raw.get("address"):
                addr = loc.raw["address"]
                country = addr.get("country", "")
                for field in ("city", "town", "village", "hamlet", "suburb", "neighbourhood", "county", "state"):
                    if field in addr:
                        city = addr[field]
                        break
                if not city:
                    city = loc.address.split(",")[0]

            city_slug = slugify(city) if city else ""
            country_slug = slugify(country) if country else ""
            for i in indices:
                resolved_names[i] = (country_slug, city_slug)
            cache.add(lat, lon, city_slug, country_slug)

            display_city = city or "[dim]unknown[/dim]"
            display_country = country or "[dim]unknown[/dim]"
            file_count = len(indices)
            console.print(f"  [green]{display_city}[/green], [cyan]{display_country}[/cyan] — {file_count} image(s)")

        # Save resolved locations to CSV for future runs
        cache.save_csv()
        console.print(f"  [dim]Saved {len(cache._entries)} location(s) to locations.csv[/dim]\n")

    # ── Phase 4: Rename files (no API calls) ────────────────────────────
    console.print(f"\n[bold]Phase 4:[/bold] Renaming files...")
    location_groups = defaultdict(list)
    for i, (img_path, _lat, _lon) in enumerate(gps_data):
        loc_key = resolved_names[i]
        if loc_key[0] or loc_key[1]:  # has country or city
            location_groups[loc_key].append(img_path)
        else:
            no_gps.append(img_path)

    renamed = 0
    for (country_slug, city_slug), paths in sorted(location_groups.items()):
        paths.sort(key=lambda p: get_date_taken(p))
        date_groups = defaultdict(list)
        for path in paths:
            dt = get_date_taken(path)
            date_groups[dt.strftime("%Y-%m-%d")].append(path)

        for date_str, date_paths in sorted(date_groups.items()):
            if len(date_paths) == 1:
                ext = os.path.splitext(date_paths[0])[1]
                new_name = f"{country_slug}-{city_slug}-{date_str}{ext}" if country_slug else f"{city_slug}-{date_str}{ext}"
                new_path = os.path.join(os.path.dirname(date_paths[0]), new_name)
                if date_paths[0] != new_path and not os.path.exists(new_path):
                    try:
                        sandbox_rename(folder, date_paths[0], new_path)
                        console.print(f"  [green]{os.path.basename(date_paths[0])}[/green] → [bold]{new_name}[/bold]")
                        log(f"Renamed: {os.path.basename(date_paths[0])} → {new_name}")
                        renamed += 1
                    except PathEscapeError as e:
                        console.print(f"  [red]BLOCKED:[/red] {e}")
            else:
                for i, path in enumerate(date_paths, 1):
                    ext = os.path.splitext(path)[1]
                    new_name = f"{country_slug}-{city_slug}-{date_str}-{i}{ext}" if country_slug else f"{city_slug}-{date_str}-{i}{ext}"
                    new_path = os.path.join(os.path.dirname(path), new_name)
                    if path != new_path and not os.path.exists(new_path):
                        try:
                            sandbox_rename(folder, path, new_path)
                            console.print(f"  [green]{os.path.basename(path)}[/green] → [bold]{new_name}[/bold]")
                            log(f"Renamed: {os.path.basename(path)} → {new_name}")
                            renamed += 1
                        except PathEscapeError as e:
                            console.print(f"  [red]BLOCKED:[/red] {e}")

    _rename_by_date(folder, no_gps, renamed_counter=None)

    table = Table(title="Geotag Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Locations resolved", str(len(unique_locations)))
    table.add_row("Renamed (GPS)", str(renamed))
    table.add_row("Renamed (date fallback)", str(len(no_gps)))
    table.add_row("Proximity radius", f"{config.geocode_radius_km} km")
    table.add_row("Source", "CSV cache" if use_existing_csv else "API")
    console.print(table)
    log(f"Geotag done: {renamed} GPS, {len(no_gps)} date-fallback")


def _rename_by_date(folder, paths, renamed_counter=None):
    count = 0
    for path in paths:
        dt = get_date_taken(path)
        ext = os.path.splitext(path)[1]
        new_name = f"{dt:%Y-%m-%d-%H-%M}{ext}"
        new_path = os.path.join(os.path.dirname(path), new_name)
        if not os.path.exists(new_path):
            try:
                sandbox_rename(folder, path, new_path)
                console.print(f"  [green]{os.path.basename(path)}[/green] → [bold]{new_name}[/bold]")
                count += 1
            except PathEscapeError as e:
                console.print(f"  [red]BLOCKED:[/red] {e}")
    return count


# ---------------------------------------------------------------------------
# Flow: Deduplication
# ---------------------------------------------------------------------------

def flow_dedupe(folder):
    check_llm()
    log(f"Dedupe started: {folder}")
    console.print(f"\n[bold]Scanning[/bold] [cyan]{folder}[/cyan]...")
    df = collect_images(folder)
    console.print(f"[bold green]Found {len(df)} images.[/bold green]\n")

    renamed = 0
    exact_groups = find_duplicate_groups(df)
    exact_count = len(exact_groups)
    if exact_count > 0:
        console.print(f"[bold yellow]{exact_count} exact duplicate group(s) found.[/bold yellow]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Reviewing exact duplicates...", total=exact_count)
            for hash_val, group in exact_groups:
                paths = group["path"].tolist()
                n = len(paths)
                progress.update(task, description=f"[cyan]Exact ({n}): {', '.join(os.path.basename(p) for p in paths)}[/cyan]")
                result = send_to_llm(paths, near_duplicate=False)
                if not result.get("is_duplicate"):
                    progress.advance(task)
                    continue
                best_name = result.get("best")
                best_path = next((p for p in paths if os.path.basename(p) == best_name), paths[0])
                to_mark = [p for p in paths if p != best_path]
                console.print(f"  [green]Duplicate group[/green] ({n} files): {result.get('reason', '')}")
                console.print(f"    Keeping: [bold]{os.path.basename(best_path)}[/bold]")
                for p in to_mark:
                    console.print(f"    Renaming: [red]{os.path.basename(p)}[/red]")
                renamed += rename_duplicates(to_mark, "__exact_dupe", "exact duplicates", os.path.basename(best_path), sandbox=folder)
                progress.advance(task)
        console.print()

    df = df[df["path"].apply(os.path.exists)].copy()
    near_groups = find_near_duplicate_groups(df)
    near_count = len(near_groups)
    if near_count > 0:
        console.print(f"[bold yellow]{near_count} near-duplicate group(s) found.[/bold yellow]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Reviewing near-duplicates...", total=near_count)
            for paths in near_groups:
                n = len(paths)
                progress.update(task, description=f"[cyan]Near ({n}): {', '.join(os.path.basename(p) for p in paths)}[/cyan]")
                result = send_to_llm(paths, near_duplicate=True)
                if not result.get("is_duplicate"):
                    progress.advance(task)
                    continue
                best_name = result.get("best")
                best_path = next((p for p in paths if os.path.basename(p) == best_name), paths[0])
                to_mark = [p for p in paths if p != best_path]
                console.print(f"  [green]Near-duplicate group[/green] ({n} files): {result.get('reason', '')}")
                console.print(f"    Keeping: [bold]{os.path.basename(best_path)}[/bold]")
                for p in to_mark:
                    console.print(f"    Renaming: [red]{os.path.basename(p)}[/red]")
                renamed += rename_duplicates(to_mark, "__near_dupe", "near-duplicates", os.path.basename(best_path), sandbox=folder)
                progress.advance(task)
        console.print()

    total_groups = exact_count + near_count
    if total_groups == 0:
        console.print("[dim]No duplicates found.[/dim]")
        return

    table = Table(title="Schmarchive Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Files renamed", str(renamed))
    console.print(table)
    log(f"Dedupe done: {renamed} files renamed")


# ---------------------------------------------------------------------------
# Flow: Move duplicates
# ---------------------------------------------------------------------------

def flow_move(folder):
    log(f"Move duplicates started: {folder}")
    files = [
        os.path.join(root, f)
        for root, _, filenames in os.walk(folder, followlinks=False)
        for f in filenames
        if f.lower().endswith(config.image_extensions)
        and ("__exact_dupe" in f or "__near_dupe" in f)
    ]
    if not files:
        console.print("[dim]No renamed duplicates found to move.[/dim]")
        return

    dest = os.path.join(os.path.dirname(os.path.abspath(folder)), config.duplicates_folder)

    console.print(f"  Source:  [cyan]{folder}[/cyan]")
    console.print(f"  Dest:    [cyan]{dest}[/cyan]")
    console.print(f"  Files:   [yellow]{len(files)}[/yellow]")
    confirm = prompt("\nMove these files? [y/N] ").strip().lower()
    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    os.makedirs(dest, exist_ok=True)
    console.print(f"\n[bold]Moving {len(files)} file(s)...[/bold]\n")
    moved = move_files(files, dest, folder)

    table = Table(title="Move Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Files moved", str(moved))
    console.print(table)
    log(f"Move done: {moved} files moved to {dest}")


# ---------------------------------------------------------------------------
# Flow: Normalize filenames
# ---------------------------------------------------------------------------

def _has_non_ascii(text):
    return any(ord(c) > 127 for c in text)


def _normalize_filename(name):
    """Replace non-ASCII chars in filename, preserving extension."""
    base, ext = os.path.splitext(name)
    normalized = base.translate(_CHAR_MAP)
    # Strip any remaining non-ASCII chars
    normalized = "".join(c for c in normalized if ord(c) < 128)
    return normalized + ext


def flow_normalize(folder):
    log(f"Normalize started: {folder}")
    files = [
        os.path.join(root, f)
        for root, _, filenames in os.walk(folder, followlinks=False)
        for f in filenames
        if f.lower().endswith(config.image_extensions)
    ]
    if not files:
        console.print("[dim]No images found.[/dim]")
        return

    to_rename = []
    for path in files:
        name = os.path.basename(path)
        if _has_non_ascii(name):
            new_name = _normalize_filename(name)
            new_path = os.path.join(os.path.dirname(path), new_name)
            if path != new_path and not os.path.exists(new_path):
                to_rename.append((path, new_path))

    if not to_rename:
        console.print("[dim]No non-ASCII filenames found.[/dim]")
        return

    console.print(f"  [yellow]{len(to_rename)}[/yellow] file(s) with non-ASCII names:\n")
    for src, dst in to_rename[:20]:
        console.print(f"  [red]{os.path.basename(src)}[/red] → [green]{os.path.basename(dst)}[/green]")
    if len(to_rename) > 20:
        console.print(f"  [dim]... and {len(to_rename) - 20} more[/dim]")

    confirm = prompt("\nRename these files? [y/N] ").strip().lower()
    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    renamed = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Normalizing filenames...", total=len(to_rename))
        for src, dst in to_rename:
            progress.update(task, description=f"[cyan]{os.path.basename(src)}[/cyan]")
            try:
                sandbox_rename(folder, src, dst)
                renamed += 1
            except PathEscapeError as e:
                console.print(f"  [red]BLOCKED:[/red] {e}")
            progress.advance(task)

    table = Table(title="Normalize Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Files renamed", str(renamed))
    console.print(table)
    log(f"Normalize done: {renamed} files renamed")


# ---------------------------------------------------------------------------
# Flow: Organize
# ---------------------------------------------------------------------------

import re

_MONTH_NAMES = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
}

_DATE_LOCATION_RE = re.compile(
    r"^(.+?)-(\d{4})-(\d{2})-(\d{2})(?:-\d+)?$"  # location-YYYY-MM-DD[-N]
)
_DATE_ONLY_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})(?:-\d{1,2}){0,3}$"  # YYYY-MM-DD[-HH-MM[-N]]
)
_CAMERA_DATE_RE = re.compile(
    r"^(?:IMG|VID|PANO|MVIMG|PXL)_?(\d{4})(\d{2})(\d{2})[_\d]*$",  # IMG_YYYYMMDD_HHMMSS
)
_COUNTRY_CITY_DATE_RE = re.compile(
    r"^(.+?)-(.+?)-(\d{4})-(\d{2})-(\d{2})(?:-\d+)?$"  # country-city-YYYY-MM-DD[-N]
)


_DUPE_SUFFIX_RE = re.compile(r"-__(?:exact|near)_dupe$")


def _parse_filename(filename):
    """Extract (year, month, country, location) from a filename. Country and location may be None."""
    base, _ext = os.path.splitext(filename)
    base = _DUPE_SUFFIX_RE.sub("", base)  # strip dupe suffixes first
    m = _COUNTRY_CITY_DATE_RE.match(base)
    if m:
        return m.group(3), m.group(4), slugify(m.group(1)), slugify(m.group(2))
    m = _DATE_ONLY_RE.match(base)
    if m:
        return m.group(1), m.group(2), None, None
    m = _DATE_LOCATION_RE.match(base)
    if m:
        return m.group(2), m.group(3), None, slugify(m.group(1))
    m = _CAMERA_DATE_RE.match(base)
    if m:
        return m.group(1), m.group(2), None, None
    return None, None, None, None


_MONTH_NAME_SET = set(_MONTH_NAMES.values())
_YEAR_RE = re.compile(r"^\d{4}$")


def _is_already_organized(path, folder):
    """Check if file is already inside a YYYY/Month[/Location] subfolder."""
    rel = os.path.relpath(path, folder)
    parts = Path(rel).parts
    if len(parts) < 3:
        return False  # root or just one subfolder — not organized yet
    if len(parts) >= 4:
        return _YEAR_RE.match(parts[0]) and parts[1] in _MONTH_NAME_SET
    return _YEAR_RE.match(parts[0]) and parts[1] in _MONTH_NAME_SET


def flow_organize_by_date_location(folder):
    log(f"Organize by date/location started: {folder}")
    files = [
        os.path.join(root, f)
        for root, _, filenames in os.walk(folder, followlinks=False)
        for f in filenames
        if f.lower().endswith(config.image_extensions)
    ]
    if not files:
        console.print("[dim]No images found.[/dim]")
        return

    # Filter out already-organized files
    already_organized = [f for f in files if _is_already_organized(f, folder)]
    files = [f for f in files if not _is_already_organized(f, folder)]
    if already_organized:
        console.print(f"  [dim]{len(already_organized)} file(s) already in YYYY/Month structure — skipped[/dim]")

    # Parse filenames
    moves: list[tuple[str, str]] = []  # (src, dst_dir)
    unparsed = []
    for path in files:
        name = os.path.basename(path)
        year, month, country, location = _parse_filename(name)
        if year and month:
            parts = [year, _MONTH_NAMES.get(month, month)]
            if country:
                parts.append(country)
            if location:
                parts.append(location)
            dest_dir = os.path.join(folder, *parts)
            moves.append((path, dest_dir))
        else:
            unparsed.append(path)

    if not moves:
        console.print("[dim]No parsable filenames found.[/dim]")
        return

    # Show preview
    dest_dirs = sorted(set(d for _, d in moves))
    console.print(f"  [yellow]{len(moves)}[/yellow] file(s) will be organized into:\n")
    for d in dest_dirs[:30]:
        count = sum(1 for _, dd in moves if dd == d)
        rel = os.path.relpath(d, folder)
        console.print(f"  [cyan]{rel}/[/cyan] ({count} files)")
    if len(dest_dirs) > 30:
        console.print(f"  [dim]... and {len(dest_dirs) - 30} more directories[/dim]")
    if unparsed:
        console.print(f"\n  [dim]{len(unparsed)} file(s) could not be parsed — skipped[/dim]")

    confirm = prompt("\nProceed? [y/N] ").strip().lower()
    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    # Create all directories first
    console.print(f"\n[bold]Creating directories...[/bold]")
    for d in dest_dirs:
        os.makedirs(d, exist_ok=True)
        console.print(f"  [green]{os.path.relpath(d, folder)}/[/green]")

    # Move files
    console.print(f"\n[bold]Moving files...[/bold]")
    moved = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Moving...", total=len(moves))
        for src, dest_dir in moves:
            progress.update(task, description=f"[cyan]{os.path.basename(src)}[/cyan]")
            try:
                src_resolved = sandbox_resolve(folder, src)
                dst = os.path.join(dest_dir, os.path.basename(src_resolved))
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(src_resolved, dst)
                moved += 1
            except PathEscapeError as e:
                console.print(f"  [red]BLOCKED:[/red] {e}")
            progress.advance(task)

    table = Table(title="Organize Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Files moved", str(moved))
    table.add_row("Directories created", str(len(dest_dirs)))
    table.add_row("Skipped (unparsed)", str(len(unparsed)))
    console.print(table)
    log(f"Organize by date/location done: {moved} files moved")


def flow_organize_by_subject(folder):
    check_llm()
    log(f"Organize by subject started: {folder}")

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith(config.image_extensions)
    ]
    if not files:
        console.print("[dim]No images found at root level.[/dim]")
        return

    categories = list(config.photo_categories)
    console.print(f"  [yellow]{len(files)}[/yellow] image(s) to categorize.")
    console.print(f"  [dim]Categories: {', '.join(categories)}[/dim]")

    confirm = prompt("\nProceed? [y/N] ").strip().lower()
    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    category_files: dict[str, list[str]] = defaultdict(list)
    failed: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Categorizing...", total=len(files))
        for path in files:
            progress.update(task, description=f"[cyan]{os.path.basename(path)}[/cyan]")
            category = _categorize_photo(path, categories)
            if category:
                category_files[category].append(path)
            else:
                failed.append(path)
            progress.advance(task)

    if not category_files:
        console.print("[dim]No images categorized.[/dim]")
        return

    console.print()
    for cat, paths in sorted(category_files.items()):
        console.print(f"  [green]{cat}[/green] — {len(paths)} image(s)")
    if failed:
        console.print(f"  [dim]{len(failed)} uncategorized[/dim]")

    confirm = prompt("\nMove files into category folders? [y/N] ").strip().lower()
    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    moved = 0
    for cat, paths in category_files.items():
        dest_dir = os.path.join(folder, cat)
        os.makedirs(dest_dir, exist_ok=True)
        for src in paths:
            try:
                src_resolved = sandbox_resolve(folder, src)
                dst = os.path.join(dest_dir, os.path.basename(src_resolved))
                shutil.move(src_resolved, dst)
                moved += 1
            except PathEscapeError as e:
                console.print(f"  [red]BLOCKED:[/red] {e}")

    table = Table(title="Organize by Subject Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Files moved", str(moved))
    table.add_row("Categories", str(len(category_files)))
    console.print(table)
    log(f"Organize by subject done: {moved} files moved")


def _categorize_photo(path, categories):
    """Ask the LLM to categorize a photo into one of the given categories."""
    import base64, io, json

    try:
        img = Image.open(path)
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

    category_list = ", ".join(f'"{c}"' for c in categories)
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": (
            f"Look at this photo and identify its main subject.\n\n"
            f"Choose EXACTLY one category from this list:\n[{category_list}]\n\n"
            f"Reply ONLY with valid JSON, no other text:\n"
            f'{{"category": "chosen_category"}}'
        )},
    ]

    try:
        r = requests.post(config.llm_url, json={
            "model": config.llm_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": 50,
        }, headers=_llm_headers(), timeout=config.llm_timeout)
        if r.status_code != 200:
            return None
        raw = r.json()["choices"][0]["message"]["content"]
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        result = json.loads(raw)
        cat = result.get("category", "").lower().strip()
        if cat in categories:
            return cat
        return None
    except Exception:
        return None


def flow_organize(folder):
    console.print()
    console.print("[bold]Organize photos by:[/bold]")
    console.print("  [cyan]1[/cyan] — Year / Month / Location")
    console.print("  [cyan]2[/cyan] — AI subject categorization")
    console.print("  [cyan]q[/cyan] — Back")

    choice = prompt("\n> ").strip().lower()
    if choice == "1":
        flow_organize_by_date_location(folder)
    elif choice == "2":
        flow_organize_by_subject(folder)
    elif choice not in ("q", "quit", "back"):
        console.print("[red]Invalid choice.[/red]")


def flow_identify(folder):
    log(f"Identify started: {folder}")
    check_llm()

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith(config.image_extensions)
    ]
    if not files:
        console.print("[dim]No images found at root level.[/dim]")
        return

    console.print(f"  [yellow]{len(files)}[/yellow] image(s) to identify by location.")
    console.print(f"  [dim]API delay: {config.geocode_delay_sec}s between requests[/dim]")

    confirm = prompt("\nProceed? [y/N] ").strip().lower()
    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    renamed = 0
    skipped = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Identifying...", total=len(files))
        for path in files:
            progress.update(task, description=f"[cyan]{os.path.basename(path)}[/cyan]")
            gps = get_gps(path)
            if not gps:
                skipped += 1
                progress.advance(task)
                continue

            lat, lon = gps
            try:
                loc = _get_geocoder().reverse((lat, lon), language="en")
                time.sleep(config.geocode_delay_sec)
            except Exception:
                failed += 1
                progress.advance(task)
                continue

            name = None
            if loc and loc.raw.get("address"):
                addr = loc.raw["address"]
                for field in ("building", "tourism", "historic", "natural"):
                    if field in addr:
                        name = addr[field]
                        break

            if not name:
                progress.advance(task)
                continue

            slug = _slugify(name)
            ext = os.path.splitext(path)[1].lower()
            new_name = f"{slug}{ext}"
            new_path = os.path.join(folder, new_name)

            if os.path.exists(new_path) and os.path.abspath(new_path) != os.path.abspath(path):
                skipped += 1
                progress.advance(task)
                continue

            try:
                os.rename(path, new_path)
                renamed += 1
            except OSError:
                failed += 1

            progress.advance(task)

    table = Table(title="Identify Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Renamed", str(renamed))
    table.add_row("Skipped (no GPS / no name / exists)", str(skipped))
    table.add_row("Failed", str(failed))
    console.print(table)
    log(f"Identify done: {renamed} renamed, {skipped} skipped, {failed} failed")


def flow_datetag(folder):
    log(f"Date-tag started: {folder}")
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith(config.image_extensions)
    ]
    if not files:
        console.print("[dim]No images found at root level.[/dim]")
        return

    console.print(f"  [yellow]{len(files)}[/yellow] image(s) to date-tag.")
    console.print("  [dim]Uses EXIF date (or file creation date as fallback).[/dim]")

    confirm = prompt("\nProceed? [y/N] ").strip().lower()
    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    renamed = 0
    skipped = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Date-tagging...", total=len(files))
        for path in files:
            progress.update(task, description=f"[cyan]{os.path.basename(path)}[/cyan]")
            try:
                dt = get_date_taken(path)
                ext = os.path.splitext(path)[1].lower()
                base_name = f"{dt:%Y-%m-%d-%H-%M}"
                new_path = os.path.join(folder, base_name + ext)

                if os.path.exists(new_path) and os.path.abspath(new_path) != os.path.abspath(path):
                    n = 1
                    while os.path.exists(os.path.join(folder, f"{base_name}-{n}{ext}")):
                        n += 1
                    new_path = os.path.join(folder, f"{base_name}-{n}{ext}")

                os.rename(path, new_path)
                renamed += 1
            except Exception:
                failed += 1

            progress.advance(task)

    table = Table(title="Date-tag Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Renamed", str(renamed))
    table.add_row("Skipped (already named / exists)", str(skipped))
    table.add_row("Failed", str(failed))
    console.print(table)
    log(f"Date-tag done: {renamed} renamed, {skipped} skipped, {failed} failed")


def flow_pluck(folder):
    check_llm()

    subject = prompt("Subject to pluck: ").strip()
    if not subject:
        console.print("[dim]Cancelled.[/dim]")
        return

    if len(subject) > 50:
        console.print("[red]Subject must be 50 characters or fewer.[/red]")
        return

    if not subject.isascii():
        console.print("[red]Subject must contain only ASCII characters.[/red]")
        return

    safe_name = subject.lower().replace(" ", "-")
    safe_name = "".join(c for c in safe_name if c.isascii() and c.isalnum() or c == "-")

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and f.lower().endswith(config.image_extensions)
    ]
    if not files:
        console.print("[dim]No images found at root level.[/dim]")
        return

    console.print(f"  [yellow]{len(files)}[/yellow] image(s) to scan for [bold]{subject}[/bold].")

    confirm = prompt("\nProceed? [y/N] ").strip().lower()
    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    matched: list[str] = []
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning...", total=len(files))
        for path in files:
            progress.update(task, description=f"[cyan]{os.path.basename(path)}[/cyan]")
            result = _pluck_match(path, subject)
            if result is True:
                matched.append(path)
                progress.console.print(f"  [green]+ DETECTED[/green] {os.path.basename(path)}")
            elif result is False:
                progress.console.print(f"  [dim]- not detected[/dim] {os.path.basename(path)}")
            else:
                failed += 1
                progress.console.print(f"  [red]! error[/red] {os.path.basename(path)}")
            progress.advance(task)

    if not matched:
        console.print("[dim]No matching images found.[/dim]")
        log(f"Pluck '{subject}': 0 matches")
        return

    console.print(f"\n  [green]{len(matched)}[/green] image(s) match [bold]{subject}[/bold]:")
    for p in matched[:20]:
        console.print(f"    {os.path.basename(p)}")
    if len(matched) > 20:
        console.print(f"    [dim]... and {len(matched) - 20} more[/dim]")

    dest_dir = os.path.join(folder, safe_name)
    confirm = prompt(f"\nMove to {safe_name}/? [y/N] ").strip().lower()
    if confirm != "y":
        console.print("[dim]Cancelled.[/dim]")
        return

    os.makedirs(dest_dir, exist_ok=True)
    moved = 0
    for src in matched:
        try:
            src_resolved = sandbox_resolve(folder, src)
            dst = os.path.join(dest_dir, os.path.basename(src_resolved))
            shutil.move(src_resolved, dst)
            moved += 1
        except PathEscapeError as e:
            console.print(f"  [red]BLOCKED:[/red] {e}")

    log(f"Pluck '{subject}': {moved} files moved to {safe_name}/")
    console.print(f"\n  [green]{moved}[/green] files moved to [bold]{safe_name}/[/bold]")


def _pluck_match(path, subject):
    """Ask the LLM if the image contains the given subject. Returns True/False/None."""
    import base64, io, json

    try:
        img = Image.open(path)
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": (
            f"Does this image contain a {subject}?\n\n"
            f"Reply ONLY with valid JSON, no other text:\n"
            f'{{"match": true}} or {{"match": false}}'
        )},
    ]

    try:
        r = requests.post(config.llm_url, json={
            "model": config.llm_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": 20,
        }, headers=_llm_headers(), timeout=config.llm_timeout)
        if r.status_code != 200:
            return None
        raw = r.json()["choices"][0]["message"]["content"]
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        result = json.loads(raw)
        return bool(result.get("match", False))
    except Exception:
        return None


def flow_config():
    from pathlib import Path
    import json

    fields = [
        ("photo_folder", "Photo Folder", str, config.photo_folder),
        ("duplicates_folder", "Duplicates Folder", str, config.duplicates_folder),
        ("llm_url", "LLM URL", str, config.llm_url),
        ("llm_model", "LLM Model", str, config.llm_model),
        ("llm_timeout", "LLM Timeout (sec)", int, config.llm_timeout),
        ("llm_api_key", "LLM API Key", str, config.llm_api_key),
        ("geocoder_user_agent", "Geocoder User Agent", str, config.geocoder_user_agent),
        ("geocoder_timeout", "Geocoder Timeout (sec)", int, config.geocoder_timeout),
        ("geocode_radius_km", "Geocode Radius (km)", float, config.geocode_radius_km),
        ("geocode_delay_sec", "Geocode Delay (sec)", float, config.geocode_delay_sec),
        ("geocode_retries", "Geocode Retries", int, config.geocode_retries),
        ("near_duplicate_threshold", "Near-Dup Threshold", int, config.near_duplicate_threshold),
        ("blur_threshold", "Blur Threshold", float, config.blur_threshold),
        ("photo_categories", "Photo Categories", list, list(config.photo_categories)),
        ("log_file", "Log File", str, config.log_file),
        ("log_max_lines", "Log Max Lines", int, config.log_max_lines),
    ]

    console.print()
    console.print("[bold]Current configuration:[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Setting", style="bold")
    table.add_column("Value", style="green")
    table.add_column("Type", style="dim")
    for i, (_, label, typ, val) in enumerate(fields, 1):
        display = ", ".join(val) if isinstance(val, list) else str(val)
        table.add_row(str(i), label, display, typ.__name__)
    console.print(table)

    console.print("\n  [dim]Enter a number to edit, or [bold]s[/bold] to save and exit.[/dim]")
    console.print("  [dim]Leave blank to keep current value.[/dim]\n")

    while True:
        choice = prompt("> ").strip().lower()
        if choice in ("s", "save", "q", "quit", ""):
            break

        try:
            idx = int(choice) - 1
            if not (0 <= idx < len(fields)):
                console.print("[red]Invalid number.[/red]")
                continue
        except ValueError:
            console.print("[red]Enter a number or 's' to save.[/red]")
            continue

        field_name, label, typ, current = fields[idx]
        display = ", ".join(current) if isinstance(current, list) else str(current)
        new_val = prompt(f"  {label} [{display}]: ").strip()

        if not new_val:
            console.print(f"  [dim]Keeping: {display}[/dim]")
            continue

        try:
            if typ is list:
                parsed = [item.strip() for item in new_val.split(",") if item.strip()]
                setattr(config, field_name, tuple(parsed))
                console.print(f"  [green]{label} = {', '.join(parsed)}[/green]")
            else:
                parsed = typ(new_val)
                setattr(config, field_name, parsed)
                console.print(f"  [green]{label} = {parsed}[/green]")
        except (ValueError, TypeError):
            console.print(f"  [red]Invalid {typ.__name__} value.[/red]")
            continue

        fields[idx] = (field_name, label, typ, getattr(config, field_name))

    # Build save dict
    save_data = {}
    for field_name, label, typ, val in fields:
        save_data[field_name] = list(val) if isinstance(val, tuple) else val

    cfg_path = Path(__file__).parent / "config.json"
    try:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)
        console.print(f"\n  [green]Saved to {cfg_path}[/green]")
        log(f"Config saved to {cfg_path}")
    except Exception as e:
        console.print(f"\n  [red]Failed to save: {e}[/red]")


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main():
    log("schmarchive started")
    console.print(Panel.fit(
        "[bold magenta]"
        "    ____       _                              _     _           \n"
        "   / ___|  ___| |__  _ __ ___   __ _ _ __ ___| |__ (_)_   _____ \n"
        "   \\___ \\ / __| '_ \\| '_ ` _ \\ / _` | '__/ __| '_ \\| \\ \\ / / _ \\\n"
        "    ___) | (__| | | | | | | | | (_| | | | (__| | | | |\\ V /  __/\n"
        "   |____/ \\___|_| |_|_| |_| |_|\\__,_|_|  \\___|_| |_|_| \\_/ \\___|\n"
        "[/bold magenta]",
        border_style="magenta",
    ))

    console.print("[dim]Deduplicate, geotag, organize, and normalize your photos[/dim]\n")

    folder = config.photo_folder
    if not os.path.isdir(folder):
        console.print(f"[red]Photo folder not found: {folder}[/red]")
        console.print("[dim]Use option 9 (Configure) to set the photo folder.[/dim]")
        return

    while True:
        console.print()
        console.print("[bold]What would you like to do?[/bold]")
        console.print()
        console.print("  [red]1[/red] — Geotag files (rename by location/date)")
        console.print("  [yellow]2[/yellow] — Date-tag files (rename by date only)")
        console.print("  [green]3[/green] — Deduplication (find duplicates via hash + LLM)")
        console.print("  [cyan]4[/cyan] — Move duplicates")
        console.print("  [blue]5[/blue] — Normalize filenames (replace non-ASCII chars)")
        console.print("  [magenta]6[/magenta] — Organize photos into subfolders")
        console.print("  [bright_red]7[/bright_red] — Identify landmarks (rename by building/touristic attraction)")
        console.print("  [bright_yellow]8[/bright_yellow] — Pluck images by subject (LLM-powered)")
        console.print("  [bright_green]9[/bright_green] — Configure settings")
        console.print("  [dim]q[/dim] — Quit")

        choice = prompt("\n> ").strip().lower()

        if choice == "1":
            flow_geotag(folder)
        elif choice == "2":
            flow_datetag(folder)
        elif choice == "3":
            flow_dedupe(folder)
        elif choice == "4":
            flow_move(folder)
        elif choice == "5":
            flow_normalize(folder)
        elif choice == "6":
            flow_organize(folder)
        elif choice == "7":
            flow_identify(folder)
        elif choice == "8":
            flow_pluck(folder)
        elif choice == "9":
            flow_config()
        elif choice in ("q", "quit", "exit"):
            log("schmarchive quit")
            console.print("[dim]Archive, schmarchive. Byechive![/dim]")
            break
        else:
            console.print("[red]Invalid choice.[/red]")


if __name__ == "__main__":
    import json as _json
    from pathlib import Path

    def _load_config():
        global config
        # 1. Try loading from config.json in project root
        cfg_path = Path(__file__).parent / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    data = _json.load(f)
                config = Config.from_dict(data)
                console.print(f"  [dim]Loaded config from {cfg_path}[/dim]")
            except Exception as e:
                console.print(f"  [yellow]Could not load config.json: {e}[/yellow]")

        # 2. Override with environment variables (SCHMARCHIVE_ prefix)
        env_map = {
            "SCHMARCHIVE_PHOTO_FOLDER": "photo_folder",
            "SCHMARCHIVE_DUPLICATES_FOLDER": "duplicates_folder",
            "SCHMARCHIVE_LLM_URL": "llm_url",
            "SCHMARCHIVE_LLM_MODEL": "llm_model",
            "SCHMARCHIVE_LLM_TIMEOUT": "llm_timeout",
            "SCHMARCHIVE_LLM_API_KEY": "llm_api_key",
            "SCHMARCHIVE_GEOCODE_RADIUS_KM": "geocode_radius_km",
            "SCHMARCHIVE_NEAR_DUPLICATE_THRESHOLD": "near_duplicate_threshold",
            "SCHMARCHIVE_BLUR_THRESHOLD": "blur_threshold",
            "SCHMARCHIVE_GEOCODER_TIMEOUT": "geocoder_timeout",
            "SCHMARCHIVE_LOG_MAX_LINES": "log_max_lines",
        }
        for env_key, field_name in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                field_type = type(getattr(config, field_name))
                try:
                    setattr(config, field_name, field_type(val))
                except (ValueError, TypeError):
                    console.print(f"  [yellow]Invalid value for {env_key}: {val}[/yellow]")

        # 3. Sync module-level log settings
        global _LOG_FILE, _LOG_MAX_LINES
        _LOG_FILE = config.log_file
        _LOG_MAX_LINES = config.log_max_lines

    _load_config()
    main()
