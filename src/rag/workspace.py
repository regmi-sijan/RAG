"""Local GUI persistence. No fetching, indexing, or model calls happen here."""
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def read_json(path: Path, fallback: Path) -> dict:
    value = json.loads((path if path.exists() else fallback).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_url(url: str, kind: str) -> str:
    url = url.strip()
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise ValueError("Enter a valid HTTPS URL with a valid hostname and port.") from error
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username
            or parsed.password or any(c.isspace() for c in url)):
        raise ValueError("Enter an HTTPS URL without credentials or whitespace.")
    if port not in {None, 443}:
        raise ValueError("Source URLs must use the standard HTTPS port 443.")
    if kind == "youtube":
        host = parsed.hostname.lower()
        if host == "youtu.be":
            video_id = parsed.path.strip("/")
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            parts = parsed.path.strip("/").split("/")
            video_id = (parse_qs(parsed.query).get("v", [""])[0]
                        if parsed.path == "/watch" else
                        parts[1] if len(parts) == 2 and parts[0] in {"shorts", "embed", "live"} else "")
        else:
            video_id = ""
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError("Enter a YouTube video URL with a valid 11-character video ID.")
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


def save_pdf(directory: Path, name: str, content: bytes) -> Path:
    if (not name or Path(name).name != name or "\\" in name or any(ord(c) < 32 for c in name)
            or not name.lower().endswith(".pdf")):
        raise ValueError("Use a plain filename ending in .pdf.")
    if len(content) > 50 * 1024 * 1024:
        raise ValueError("PDF exceeds the 50 MB upload limit.")
    if not content.startswith(b"%PDF-"):
        raise ValueError("The uploaded file does not have a PDF header.")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    # Publish only a complete file, without overwriting an existing name/symlink.
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=".upload-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def list_pdfs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*")
                  if p.is_file() and not p.is_symlink() and p.suffix.lower() == ".pdf"
                  and p.resolve().is_relative_to(directory.resolve()))


def remove_pdf(root: Path, path: Path) -> Path:
    """Remove from the searchable library while retaining a recoverable copy."""
    root, path = Path(root).resolve(), Path(path)
    library = root / "data/pdfs"
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".pdf" or not path.resolve().is_relative_to(library.resolve()):
        raise ValueError("Choose a PDF inside the project PDF library.")
    relative = path.resolve().relative_to(library.resolve())
    destination = root / "data/removed-pdfs" / uuid.uuid4().hex / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.rename(destination)
    return destination


def restore_pdf(root: Path, path: Path) -> Path:
    root, path = Path(root).resolve(), Path(path)
    trash = root / "data/removed-pdfs"
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(trash.resolve()):
        raise ValueError("Choose a removed PDF from this project.")
    parts = path.resolve().relative_to(trash.resolve()).parts
    if len(parts) < 2 or path.suffix.lower() != ".pdf":
        raise ValueError("Invalid removed PDF location.")
    target = root / "data/pdfs" / Path(*parts[1:])
    if not target.parent.resolve().is_relative_to((root / "data/pdfs").resolve()):
        raise ValueError("Restore destination must be inside the PDF library.")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive publication protects any newly uploaded file with the same name.
    os.link(path, target)
    path.unlink()
    return target
