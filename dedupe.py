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
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    lm_studio_url: str = "http://localhost:1234/v1/chat/completions"
    lm_studio_model: str = "qwen/qwen2.5-vl-7b"
    lm_studio_timeout: int = 60

    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")

    geocoder_user_agent: str = "dedupe_photo_tool"
    geocoder_timeout: int = 10
    geocode_radius_km: float = 1.0
    geocode_delay_sec: float = 1.1
    geocode_retries: int = 3

    near_duplicate_threshold: int = 8
    blur_threshold: float = 50.0

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


config = Config()


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

    def get(self, lat: float, lon: float) -> Optional[str]:
        for elat, elon, name in self._entries:
            if self._haversine_km(lat, lon, elat, elon) <= self.radius_km:
                return name
        return None

    def add(self, lat: float, lon: float, name: str):
        self._entries.append((lat, lon, name))

    def _load_csv(self):
        """Load previously resolved locations from CSV."""
        try:
            with open(self.csv_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("lat"):
                        continue
                    parts = line.split(",", 2)
                    if len(parts) == 3:
                        try:
                            lat, lon, name = float(parts[0]), float(parts[1]), parts[2]
                            self._entries.append((lat, lon, name))
                        except ValueError:
                            continue
        except FileNotFoundError:
            pass

    def save_csv(self):
        """Persist all resolved locations to CSV."""
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("lat,lon,name\n")
            for lat, lon, name in self._entries:
                f.write(f"{lat:.6f},{lon:.6f},{name}\n")

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


_TURKISH_MAP = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
    "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U",
})


def slugify(text):
    text = text.lower().strip().translate(_TURKISH_MAP)
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
        d, m, s = val
        return float(d) + float(m) / 60 + float(s) / 3600

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
    cached = cache.get(lat, lon)
    if cached is not None:
        if cached:
            console.print(f"    [dim](cached: [bold]{cached}[/bold])[/dim]")
        return cached

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
        r = requests.post(config.lm_studio_url, json={
            "model": config.lm_studio_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": 200,
        }, timeout=config.lm_studio_timeout)
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

def check_lm_studio():
    models_url = config.lm_studio_url.replace("/chat/completions", "/models")
    try:
        r = requests.get(models_url, timeout=5)
        r.raise_for_status()
        models = [m.get("id", "") for m in r.json().get("data", [])]
        if not models:
            console.print("[red]No models loaded in LM Studio.[/red]")
            raise SystemExit(1)
        console.print(f"  LM Studio: [green]connected[/green] — model: {models[0]}")
    except requests.ConnectionError:
        console.print(f"[red]Cannot connect to LM Studio at {models_url}.[/red]")
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]LM Studio check failed — {e}[/red]")
        raise SystemExit(1)


def flow_geotag(folder):
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

    # ── Phase 2: Test API availability ───────────────────────────────────
    console.print("[bold]Phase 2:[/bold] Testing geocoder API...")
    if not test_geocoder():
        console.print("[red]Geocoder API is not responding.[/red]")
        console.print("[dim]Try again later or check your network.[/dim]")
        return
    console.print("  [green]API is alive[/green]\n")

    # ── Phase 3: Resolve unique locations ────────────────────────────────
    console.print("[bold]Phase 3:[/bold] Resolving location names...")
    resolved_names: dict[int, str] = {}  # index -> name

    import time
    for idx, (lat, lon, indices) in enumerate(unique_locations):
        loc = _get_geocoder().reverse((lat, lon), language="en")
        time.sleep(config.geocode_delay_sec)

        name = ""
        if loc and loc.raw.get("address"):
            addr = loc.raw["address"]
            for field in ("city", "town", "village", "hamlet", "suburb", "neighbourhood", "county", "state"):
                if field in addr:
                    name = addr[field]
                    break
            if not name:
                name = loc.address.split(",")[0]

        slug = slugify(name) if name else ""
        for i in indices:
            resolved_names[i] = slug
        cache.add(lat, lon, slug)

        display_name = name or "[dim]unknown[/dim]"
        file_count = len(indices)
        console.print(f"  [green]{display_name}[/green] — {file_count} image(s)")

    # Save resolved locations to CSV for future runs
    cache.save_csv()
    console.print(f"  [dim]Saved {len(cache._entries)} location(s) to locations.csv[/dim]\n")

    # ── Phase 4: Rename files (no API calls) ────────────────────────────
    console.print(f"\n[bold]Phase 4:[/bold] Renaming files...")
    location_groups = defaultdict(list)
    for i, (img_path, _lat, _lon) in enumerate(gps_data):
        slug = resolved_names[i]
        if slug:
            location_groups[slug].append(img_path)
        else:
            no_gps.append(img_path)

    renamed = 0
    for loc_slug, paths in sorted(location_groups.items()):
        paths.sort(key=lambda p: get_date_taken(p))
        date_groups = defaultdict(list)
        for path in paths:
            dt = get_date_taken(path)
            date_groups[dt.strftime("%Y-%m-%d")].append(path)

        for date_str, date_paths in sorted(date_groups.items()):
            if len(date_paths) == 1:
                ext = os.path.splitext(date_paths[0])[1]
                new_name = f"{loc_slug}-{date_str}{ext}"
                new_path = os.path.join(os.path.dirname(date_paths[0]), new_name)
                if date_paths[0] != new_path and not os.path.exists(new_path):
                    try:
                        sandbox_rename(folder, date_paths[0], new_path)
                        console.print(f"  [green]{os.path.basename(date_paths[0])}[/green] → [bold]{new_name}[/bold]")
                        renamed += 1
                    except PathEscapeError as e:
                        console.print(f"  [red]BLOCKED:[/red] {e}")
            else:
                for i, path in enumerate(date_paths, 1):
                    ext = os.path.splitext(path)[1]
                    new_name = f"{loc_slug}-{date_str}-{i}{ext}"
                    new_path = os.path.join(os.path.dirname(path), new_name)
                    if path != new_path and not os.path.exists(new_path):
                        try:
                            sandbox_rename(folder, path, new_path)
                            console.print(f"  [green]{os.path.basename(path)}[/green] → [bold]{new_name}[/bold]")
                            renamed += 1
                        except PathEscapeError as e:
                            console.print(f"  [red]BLOCKED:[/red] {e}")

    _rename_by_date(folder, no_gps, renamed_counter=None)

    table = Table(title="Geotag Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Unique locations resolved", str(len(unique_locations)))
    table.add_row("Renamed (GPS)", str(renamed))
    table.add_row("Renamed (date fallback)", str(len(no_gps)))
    table.add_row("Proximity radius", f"{config.geocode_radius_km} km")
    console.print(table)


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
# Flow: Dedupe
# ---------------------------------------------------------------------------

def flow_dedupe(folder):
    check_lm_studio()
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

    table = Table(title="Dedupe Summary", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Files renamed", str(renamed))
    console.print(table)


# ---------------------------------------------------------------------------
# Flow: Move deduped
# ---------------------------------------------------------------------------

def flow_move(folder):
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

    dest = os.path.join(os.path.dirname(os.path.abspath(folder)), "duplicates")

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


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main():
    console.print(Panel.fit(
        "[bold]Photo Archive Tool[/bold]\n"
        "[dim]Deduplicate, geotag, and organize your photos[/dim]",
        border_style="cyan",
    ))

    folder = ask_folder()

    while True:
        console.print()
        console.print("[bold]What would you like to do?[/bold]")
        console.print("  [cyan]1[/cyan] — Geotag files (rename by location/date)")
        console.print("  [cyan]2[/cyan] — Dedupe (find duplicates via hash + LLM)")
        console.print("  [cyan]3[/cyan] — Move deduped files to ./duplicates")
        console.print("  [cyan]q[/cyan] — Quit")

        choice = prompt("\n> ").strip().lower()

        if choice == "1":
            flow_geotag(folder)
        elif choice == "2":
            flow_dedupe(folder)
        elif choice == "3":
            flow_move(folder)
        elif choice in ("q", "quit", "exit"):
            console.print("[dim]Bye.[/dim]")
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

        # 2. Override with environment variables (DEDUPE_ prefix)
        env_map = {
            "DEDUPE_LM_STUDIO_URL": "lm_studio_url",
            "DEDUPE_LM_STUDIO_MODEL": "lm_studio_model",
            "DEDUPE_LM_STUDIO_TIMEOUT": "lm_studio_timeout",
            "DEDUPE_GEOCODE_RADIUS_KM": "geocode_radius_km",
            "DEDUPE_NEAR_DUPLICATE_THRESHOLD": "near_duplicate_threshold",
            "DEDUPE_BLUR_THRESHOLD": "blur_threshold",
            "DEDUPE_GEOCODER_TIMEOUT": "geocoder_timeout",
        }
        for env_key, field_name in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                field_type = type(getattr(config, field_name))
                try:
                    setattr(config, field_name, field_type(val))
                except (ValueError, TypeError):
                    console.print(f"  [yellow]Invalid value for {env_key}: {val}[/yellow]")

    _load_config()
    main()
