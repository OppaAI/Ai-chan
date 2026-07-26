"""
toolkit/cache.py

Generic thread-safe TTL cache with LRU eviction.

This module provides a reusable caching layer for expensive I/O operations
across the agentic toolkit: web search, page fetching, embeddings, API calls,
robots.txt parsing, etc. Each cache instance maintains its own TTL and
eviction policy, allowing callers to tune retention independently per use case.

DESIGN PRINCIPLES:

  1. Thread-safe by default
     All public methods acquire a lock for the entire operation (atomic).
     Safe for concurrent reads/writes from multiple threads without external
     locking.

  2. LRU eviction by timestamp
     When max_entries is reached, the oldest (by insertion/update time) entry
     is dropped to make room. This is FIFO-like but respects actual monotonic
     time, not insertion order, so clock skew won't break it.

  3. Lazy expiration
     TTL is checked only on access (get), not by a background reaper. Expired
     entries remain in memory until explicitly read; this trades memory for
     CPU and is safe for the ~256-entry caches typical in this toolkit.
     If you cache millions of items, add a reaper thread.

  4. Per-instance TTL + capacity
     Callers can tune each cache independently — e.g., web_search results
     cached for 15 min but fetch results for 1 hour, both from the same
     TTLCache class.

TYPICAL USAGE PATTERNS:

  Pattern 1: Exact key match (search results)
    cache_key = f"{query}|{max_results}|{pageno}"
    cached = search_cache.get(cache_key)
    if cached:
        return cached
    results = do_search(query)
    search_cache.set(cache_key, results)

  Pattern 2: URL-based (fetch results)
    cache_key = f"{url}|{max_chars}"
    cached = fetch_cache.get(cache_key)
    if cached:
        return cached
    text = extract_page(url)
    fetch_cache.set(cache_key, text)

  Pattern 3: Multiple independent caches
    embeddings_cache = TTLCache(ttl_seconds=3600, max_entries=1024)
    robots_cache = TTLCache(ttl_seconds=86400, max_entries=512)
    search_cache = TTLCache(ttl_seconds=900, max_entries=256)

CONFIGURATION:

  Via environment variables (websurf.py pattern):
    TOOLS_CACHE_TTL_SECONDS (default 900)     — 15 minutes
    TOOLS_CACHE_MAX_ENTRIES (default 256)     — entry cap before LRU eviction

  Via direct constructor:
    cache = TTLCache(ttl_seconds=3600, max_entries=512)

LIMITATIONS & NOTES:

  - NOT persisted across process restarts (in-process only)
  - No background eviction; old entries stay until accessed or LRU'd out
  - No cache statistics (hit rate, etc.) — add if needed for profiling
  - Key is always a string; values are untyped (Any)
  - Thread-safe but NOT lock-free; high contention can serialize under load
  - Timestamp is monotonic (not wall-clock), immune to clock skew

FUTURE ENHANCEMENTS:

  - Add .clear() method to flush on demand
  - Add .stats() returning (hits, misses, evictions, current_size)
  - Optional background reaper for high-volume caches
  - Typed cache wrapper (TTLCache[T]) if TypeVar-based value types matter
  - Optional write-through to persistent storage (Redis, SQLite, etc.)
"""

from __future__ import annotations

import time
import threading
from typing import Any, TypeVar

T = TypeVar("T")


class TTLCache:
    """Thread-safe cache with time-to-live expiration and LRU eviction.

    Each entry is stored with a monotonic timestamp and automatically expired
    on access if the current time exceeds (timestamp + ttl_seconds). When the
    cache reaches max_entries, the oldest entry is evicted to make room.

    All public methods (_get, set) are atomic; multiple threads can safely
    read and write concurrently.

    Attributes:
        ttl_seconds (int): Time-to-live in seconds for each cached entry.
                           Default 900 (15 minutes). Entries older than this
                           are expired on next access.
        max_entries (int): Maximum number of entries before LRU eviction.
                           Default 256. When this limit is reached, the oldest
                           (by timestamp) entry is removed.

    Example:
        >>> cache = TTLCache(ttl_seconds=900, max_entries=256)
        >>> cache.set("query_key", ["result1", "result2"])
        >>> results = cache.get("query_key")
        >>> if results:
        ...     print(results)
        ['result1', 'result2']

    Thread Safety:
        All methods are protected by an internal lock (_lock). Concurrent
        calls from multiple threads are safe without external synchronization.
    """

    def __init__(self, ttl_seconds: int = 900, max_entries: int = 256):
        """Initialize a new TTL cache.

        Args:
            ttl_seconds (int, optional):
                Time-to-live in seconds for each entry. Default 900 (15 min).
                Entries not accessed within this window are considered expired
                and removed on next access or during eviction.

            max_entries (int, optional):
                Maximum number of entries allowed before LRU eviction.
                Default 256. When the cache reaches this size, the oldest
                entry (by insertion/update timestamp) is removed to make room
                for new entries.

        Example:
            >>> search_cache = TTLCache(ttl_seconds=900, max_entries=256)
            >>> fetch_cache = TTLCache(ttl_seconds=3600, max_entries=512)
            >>> embeddings_cache = TTLCache(ttl_seconds=86400, max_entries=1024)
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value if it exists and has not expired.

        Checks the cache for the given key. If found and the entry's
        timestamp is within the TTL window (now - timestamp < ttl_seconds),
        the value is returned. If the entry has expired, it is removed from
        the cache and None is returned. If the key doesn't exist, None is
        returned.

        Thread-safe. Acquires internal lock for the entire operation.

        Args:
            key (str):
                The cache key to look up. Typically formatted as
                "{resource}|{param1}|{param2}" — e.g., "https://example.com|4000"
                or "python|5|1" for a paginated search.

        Returns:
            Any | None:
                The cached value (typically a list, string, or dict) if found
                and not expired. Returns None if the key doesn't exist or the
                entry has expired. Expired entries are removed from the cache
                on access.

        Example:
            >>> cache = TTLCache(ttl_seconds=900)
            >>> cache.set("search_key", ["result1", "result2"])
            >>> results = cache.get("search_key")
            >>> if results:
            ...     print(f"Cache hit: {results}")
            ...
            >>> missing = cache.get("nonexistent_key")
            >>> print(missing)  # None

        Notes:
            - Expiration is lazy; old entries are only removed when accessed
              or when they become victims of LRU eviction.
            - Thread-safe under concurrent access.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            ts, value = entry
            # Check if entry has exceeded TTL
            if time.monotonic() - ts > self.ttl_seconds:
                self._cache.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        """Store or update a value in the cache with current timestamp.

        Inserts the key-value pair into the cache with the current monotonic
        timestamp. If the key already exists, the value and timestamp are
        updated (effectively resetting the TTL clock for that entry).

        If the cache is at max_entries capacity, the oldest entry (by
        timestamp) is evicted to make room for the new entry.

        Thread-safe. Acquires internal lock for the entire operation.

        Args:
            key (str):
                The cache key. Typically formatted as "{resource}|{param}[|...]"
                — e.g., "https://example.com|4000" for page fetches or
                "python nlp|5|1" for search results.

            value (Any):
                The value to cache. Can be any picklable Python object:
                lists (search results), strings (fetched text), dicts
                (parsed metadata), etc. No type checking is performed.

        Returns:
            None

        Example:
            >>> cache = TTLCache(ttl_seconds=900, max_entries=256)
            >>> cache.set("search:python|5|1", [{"title": "...", "url": "..."}])
            >>> cache.set("fetch:https://example.com|4000", "extracted text...")
            >>> cache.set("embed:sentence_123", array([...]))

        Side Effects:
            - Updates the entry's timestamp to the current monotonic time.
            - If cache is full (len >= max_entries), evicts the oldest entry.
            - Does not check or remove expired entries; expiration is lazy
              (on next get() call).

        Notes:
            - Overwrites existing values silently (no warning if key exists).
            - Timestamp is updated on every set, resetting the TTL countdown.
            - Thread-safe under concurrent access.
            - LRU eviction is by timestamp, not by access pattern — least
              recently *updated* entries are evicted, not least recently *used*.
        """
        with self._lock:
            # Evict oldest entry if cache is full
            if len(self._cache) >= self.max_entries:
                oldest_key = min(
                    self._cache,
                    key=lambda k: self._cache[k][0],  # Sort by timestamp
                    default=None
                )
                if oldest_key is not None:
                    self._cache.pop(oldest_key, None)
            # Store value with current monotonic timestamp
            self._cache[key] = (time.monotonic(), value)

    def __len__(self) -> int:
        """Return the current number of entries in the cache (approximate).

        Note: This count may include expired entries, since expiration is lazy.
        The actual number of valid (non-expired) entries may be lower.

        Returns:
            int: Current size of the internal cache dict.
        """
        with self._lock:
            return len(self._cache)

    def __repr__(self) -> str:
        """Return a string representation of the cache."""
        with self._lock:
            size = len(self._cache)
        return f"TTLCache(ttl={self.ttl_seconds}s, max={self.max_entries}, entries={size})"