"""Image validation, canonicalisation and (near-)duplicate detection."""
from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 100_000_000   # decompression-bomb guard (PIL raises above 2x this)


class BadImage(Exception):
    pass


def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = g.tobytes()
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def process_image(src: Path, dst: Path, min_side: int, max_side: int, max_aspect: float) -> tuple[str, int]:
    """Validate + re-encode as RGB JPEG (EXIF-rotated, downsized). Returns (sha256 of output, dhash).
    Raises BadImage for corrupt files, tiny images, or extreme aspect ratios."""
    try:
        with Image.open(src) as im:
            im.verify()
        with Image.open(src) as im:                       # verify() invalidates the handle
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            if min(w, h) < min_side:
                raise BadImage(f"too small {w}x{h}")
            if max(w, h) / min(w, h) > max_aspect:
                raise BadImage(f"aspect ratio {w}x{h}")
            if max(w, h) > max_side:
                im.thumbnail((max_side, max_side), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, "JPEG", quality=92)
            h64 = dhash(im)
    except BadImage:
        raise
    except Exception as e:                                # truncated/corrupt/unsupported
        raise BadImage(f"unreadable: {e}") from e
    return hashlib.sha256(Path(dst).read_bytes()).hexdigest(), h64


class NearDupIndex:
    """Hamming-radius search on 64-bit dHashes. Pigeonhole: if hamming <= r < 4 then at
    least one of the four 16-bit bands is identical, so we only compare within band buckets."""

    def __init__(self, radius: int = 3):
        assert 0 <= radius < 4, "radius must be < 4 for the 4-band index to be exact"
        self.radius = radius
        self.bands: list[dict[int, list[int]]] = [{} for _ in range(4)]

    def add_if_new(self, h: int) -> bool:
        """True if h was novel (and is now indexed); False if a near-duplicate exists."""
        keys = [(h >> (16 * i)) & 0xFFFF for i in range(4)]
        for i, k in enumerate(keys):
            for other in self.bands[i].get(k, ()):
                if hamming(h, other) <= self.radius:
                    return False
        for i, k in enumerate(keys):
            self.bands[i].setdefault(k, []).append(h)
        return True
