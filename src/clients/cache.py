"""
Local cache for extraction results.

Stores extraction results as JSON files indexed by PDF hash.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union


class ExtractionCache:
    """
    File-based cache for PDF extraction results.

    Each cached result is stored as a JSON file named with the PDF's SHA-256 hash.
    Includes metadata about when the result was cached and source information.
    """

    def __init__(self, cache_dir: Union[str, Path] = "cache"):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, pdf_hash: str) -> Path:
        """Get the file path for a cached result."""
        return self.cache_dir / f"{pdf_hash}.json"

    def get(self, pdf_hash: str) -> Optional[dict]:
        """
        Retrieve a cached extraction result.

        Args:
            pdf_hash: SHA-256 hash of the PDF

        Returns:
            Cached data dictionary or None if not found
        """
        cache_path = self._get_cache_path(pdf_hash)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                return cached.get("data")
        except (json.JSONDecodeError, KeyError):
            # Corrupted cache file, remove it
            cache_path.unlink(missing_ok=True)
            return None

    def set(
        self,
        pdf_hash: str,
        data: Any,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store an extraction result in the cache.

        Args:
            pdf_hash: SHA-256 hash of the PDF
            data: Extracted data to cache (must be JSON-serializable)
            metadata: Optional metadata (source type, filename, etc.)
        """
        cache_path = self._get_cache_path(pdf_hash)

        cache_entry = {
            "data": data,
            "cached_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_entry, f, ensure_ascii=False, indent=2, default=str)

    def exists(self, pdf_hash: str) -> bool:
        """Check if a result is cached."""
        return self._get_cache_path(pdf_hash).exists()

    def invalidate(self, pdf_hash: str) -> bool:
        """
        Remove a cached result.

        Args:
            pdf_hash: SHA-256 hash of the PDF

        Returns:
            True if a cache entry was removed, False if it didn't exist
        """
        cache_path = self._get_cache_path(pdf_hash)

        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear(self) -> int:
        """
        Remove all cached results.

        Returns:
            Number of cache entries removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count

    def get_metadata(self, pdf_hash: str) -> Optional[dict]:
        """
        Get metadata for a cached result without loading the full data.

        Args:
            pdf_hash: SHA-256 hash of the PDF

        Returns:
            Metadata dictionary or None if not cached
        """
        cache_path = self._get_cache_path(pdf_hash)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                return {
                    "cached_at": cached.get("cached_at"),
                    **cached.get("metadata", {}),
                }
        except (json.JSONDecodeError, KeyError):
            return None

    def list_cached(self) -> list[dict]:
        """
        List all cached entries with their metadata.

        Returns:
            List of dictionaries with hash and metadata for each cached entry
        """
        entries = []

        for cache_file in self.cache_dir.glob("*.json"):
            pdf_hash = cache_file.stem
            metadata = self.get_metadata(pdf_hash)

            if metadata:
                entries.append({"hash": pdf_hash, **metadata})

        return sorted(entries, key=lambda x: x.get("cached_at", ""), reverse=True)
