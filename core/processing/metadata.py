"""Metadata extraction and enrichment engine for media files.

This module provides the `MetadataEngine`, which is responsible for:
- Parsing media filenames into structured metadata.
- Inferring media type from file paths and naming conventions.
- Cleaning noisy release tags from filenames using pattern-based rules.
- Enriching metadata from external providers such as TMDB and OMDb,
  with graceful fallback behavior when APIs or keys are unavailable.

The module is designed to be production-ready, dependency-light, and
safe to use in batch media scanning pipelines where partial metadata
is preferable to failure.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

from core.schemas import MediaMetadata, MediaType


class MetadataEngine:
    """Extract semantic metadata from filenames and external metadata APIs.

    This engine parses media filenames into a structured :class:`MediaMetadata`
    instance and optionally enriches it using online metadata providers such as
    TMDB and OMDb.

    Online enrichment is attempted only for video-like assets and is best-effort:
    if all providers fail or are misconfigured, the engine still returns a valid
    :class:`MediaMetadata` object with ``is_processed`` set to ``False``.
    """

    TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/multi"
    TMDB_MOVIE_CREDITS_URL = "https://api.themoviedb.org/3/movie/{id}/credits"
    TMDB_TV_CREDITS_URL = "https://api.themoviedb.org/3/tv/{id}/credits"

    OMDB_URL = "https://www.omdbapi.com/"

    DEFAULT_TIMEOUT = 5.0

    # Common media file extensions for lightweight type inference
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}
    AUDIO_EXTENSIONS = {".mp3", ".aac", ".flac", ".wav", ".ogg", ".m4a"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    def __init__(
        self,
        tmdb_api_key: str | None = None,
        omdb_api_key: str | None = None,
        junk_tag_patterns: Iterable[str] | None = None,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the metadata engine.

        API keys may be passed explicitly or loaded from environment variables:

        - TMDB: ``TMDB_API_KEY``
        - OMDb: ``OMDB_API_KEY``

        Args:
            tmdb_api_key: API key for TMDB. If ``None``, falls back to
                ``TMDB_API_KEY`` environment variable.
            omdb_api_key: API key for OMDb. If ``None``, falls back to
                ``OMDB_API_KEY`` environment variable.
            junk_tag_patterns: Optional iterable of regular expression patterns
                used to strip "junk" release tags from filenames (e.g. quality,
                codec, group names). If not provided, a reasonable default set of
                patterns is used.
            session: Optional shared :class:`requests.Session` for HTTP calls.
                If ``None``, one is created internally.
        """
        self.tmdb_key = tmdb_api_key or os.getenv("TMDB_API_KEY")
        self.omdb_key = omdb_api_key or os.getenv("OMDB_API_KEY")

        if junk_tag_patterns is None:
            junk_tag_patterns = self._default_junk_tag_patterns()

        # Compile once for efficiency and to avoid hard-coding individual tokens
        self._junk_regexes = [
            re.compile(p, flags=re.IGNORECASE) for p in junk_tag_patterns
        ]

        self._session = session or requests.Session()

    def identify(
        self,
        file_path: Path,
        user_hint: MediaType = MediaType.UNKNOWN,
        enable_online_lookup: bool = True,
    ) -> MediaMetadata:
        """Identify media metadata from a file path.

        Args:
            file_path: Path to the media file.
            user_hint: Optional user-provided hint for media type. If not
                ``MediaType.UNKNOWN``, it takes precedence over auto-detection.
            enable_online_lookup: If ``True``, attempts to enrich metadata using
                configured online providers (TMDB, OMDb). If no provider is
                configured, enrichment is skipped and the result is purely
                filename-based.

        Returns:
            A :class:`MediaMetadata` instance populated from the filename and,
            when available, from online metadata providers.
        """
        filename = file_path.stem

        title, year = self._parse_filename(filename)
        media_type = self._infer_media_type(file_path, user_hint, title, year)

        meta = MediaMetadata(
            title=title or file_path.stem,
            year=year,
            media_type=media_type,
        )

        if enable_online_lookup and media_type in [MediaType.MOVIE, MediaType.TV]:
            enriched = self._enrich_from_providers(title, year)
            if enriched:
                meta.cast = enriched.get("cast", []) or meta.cast
                meta.plot_summary = enriched.get("overview") or meta.plot_summary
                meta.is_processed = True

        return meta

    def _infer_media_type(
        self,
        file_path: Path,
        user_hint: MediaType,
        title: str | None,
        year: int | None,
    ) -> MediaType:
        """Infer media type from user hints, filename patterns, and file extension."""
        # 1. User hint always wins
        if user_hint != MediaType.UNKNOWN:
            return user_hint

        filename = file_path.name

        # 2. TV pattern must be checked first (many shows include years)
        if re.search(r"[sS]\d{2}[eE]\d{2}", filename):
            return MediaType.TV

        # 3. Movie pattern: title + year
        if title and year:
            return MediaType.MOVIE

        # 4. Extension-based fallback
        suffix = file_path.suffix.lower()
        if suffix in self.VIDEO_EXTENSIONS:
            return MediaType.VIDEO
        if suffix in self.AUDIO_EXTENSIONS:
            return MediaType.AUDIO
        if suffix in self.IMAGE_EXTENSIONS:
            return MediaType.IMAGE

        # 5. Final fallback
        return MediaType.PERSONAL

    def _parse_filename(self, filename: str) -> tuple[str, int | None]:
        """Parse a filename into a cleaned title and optional year."""
        clean_name = filename.replace(".", " ").replace("_", " ")

        year_match = re.search(r"\((\d{4})\)", clean_name) or re.search(
            r"\b(19\d{2}|20\d{2})\b", clean_name
        )

        year: int | None = None
        title_part = clean_name

        if year_match:
            year = int(year_match.group(1))
            title_part = clean_name[: year_match.start()]

        title = self._clean_title(title_part)
        return title, year

    def _clean_title(self, raw_title: str) -> str:
        """Remove junk tags, bracketed content, and excess whitespace from a title."""
        # Strip common bracketed groups like [YTS], (1080p), etc.
        raw_title = re.sub(r"\[.*?\]", " ", raw_title)
        raw_title = re.sub(r"\(.*?\)", " ", raw_title)

        # Remove generic "junk" patterns (resolutions, codecs, release tags)
        for pattern in self._junk_regexes:
            raw_title = pattern.sub(" ", raw_title)

        # Collapse whitespace and trim punctuation
        cleaned = re.sub(r"\s+", " ", raw_title).strip(" -_.")
        return cleaned

    def _default_junk_tag_patterns(self) -> list[str]:
        """Return default regex patterns for typical release 'junk' tags.

        These are intentionally pattern-based (resolutions, codecs, release
        flags, etc.) instead of a hard-coded list of specific words so that
        they generalize better to new releases.
        """
        return [
            r"\b\d{3,4}p\b",  # 720p, 1080p, 2160p, etc.
            r"\b\d{3,4}x\d{3,4}\b",  # 1920x1080, etc.
            r"\b(?:x|h|hevc)?26[45]\b",  # x264, h264, x265, hevc
            r"\b(?:hdrip|webrip|web[-_. ]?dl|bluray|b[dr]rip|dvdrip)\b",
            r"\b(?:dts|aac|ac3|flac|ddp? ?5[.\s]?1)\b",
            r"\b(?:repack|proper|extended|remastered|unrated|uncut)\b",
            r"\b(?:esub[s]?|subs?)\b",
            r"\b(?:multi|dual[ -]?audio)\b",
            r"\b(?:yts|yify|rarbg|evo|hive|lol|ettv|ntb)\b",
            r"\b(?:hc)?camrip\b",
            r"\b(?:hdr|dolby[ -]?vision|atmos)\b",
        ]

    def _enrich_from_providers(
        self,
        title: str | None,
        year: int | None,
    ) -> Mapping[str, Any] | None:
        """Try TMDB first, then OMDb, returning normalized metadata."""
        if not title:
            return None

        # Try TMDB if key configured
        if self.tmdb_key:
            tmdb_data = self._fetch_tmdb(title, year)
            if tmdb_data:
                return tmdb_data

        # Fallback to OMDb if key configured
        if self.omdb_key:
            omdb_data = self._fetch_omdb(title, year)
            if omdb_data:
                return omdb_data

        # No provider configured or all failed
        return None

    def _fetch_tmdb(self, title: str, year: int | None) -> Mapping[str, Any] | None:
        """Fetch metadata from TMDB using multi-search and credits.

        Args:
            title: Title of the media.
            year: Optional release year to improve matching.

        Returns:
            A mapping with at least ``overview`` and ``cast`` keys, or ``None``
            if the lookup fails or returns no usable results.
        """
        params: dict[str, Any] = {
            "api_key": self.tmdb_key,
            "query": title,
            "include_adult": False,
        }
        if year is not None:
            params["year"] = year

        try:
            resp = self._session.get(
                self.TMDB_SEARCH_URL, params=params, timeout=self.DEFAULT_TIMEOUT
            )
            resp.raise_for_status()
        except requests.RequestException:
            return None

        payload = resp.json()
        results = payload.get("results") or []
        if not results:
            return None

        best = results[0]
        media_type = best.get("media_type")
        tmdb_id = best.get("id")

        if not tmdb_id:
            return None

        overview = best.get("overview")

        cast: list[str] = []
        credits_url: str | None = None

        if media_type == "movie":
            credits_url = self.TMDB_MOVIE_CREDITS_URL.format(id=tmdb_id)
        elif media_type == "tv":
            credits_url = self.TMDB_TV_CREDITS_URL.format(id=tmdb_id)

        if credits_url:
            try:
                credits_resp = self._session.get(
                    credits_url,
                    params={"api_key": self.tmdb_key},
                    timeout=self.DEFAULT_TIMEOUT,
                )
                credits_resp.raise_for_status()
                credits_payload = credits_resp.json()
                cast_items = credits_payload.get("cast") or []
                cast = [c.get("name") for c in cast_items[:5] if c.get("name")]
            except requests.RequestException:
                cast = []

        return {"overview": overview, "cast": cast}

    def _fetch_omdb(self, title: str, year: int | None) -> Mapping[str, Any] | None:
        """Fetch metadata from OMDb as a fallback provider.

        Args:
            title: Title of the media.
            year: Optional release year.

        Returns:
            A mapping with at least ``overview`` and ``cast`` keys, or ``None``
            if the lookup fails or returns no usable results.
        """
        params: dict[str, Any] = {
            "apikey": self.omdb_key,
            "t": title,
        }
        if year is not None:
            params["y"] = year

        try:
            resp = self._session.get(
                self.OMDB_URL, params=params, timeout=self.DEFAULT_TIMEOUT
            )
            resp.raise_for_status()
        except requests.RequestException:
            return None

        data = resp.json()
        if data.get("Response") != "True":
            return None

        overview = data.get("Plot") or None
        cast_str = data.get("Actors") or ""
        cast = [name.strip() for name in cast_str.split(",") if name.strip()]

        return {"overview": overview, "cast": cast}
