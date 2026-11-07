
import sys
import glob
from pathlib import Path

import numpy as np
from numba import njit, prange
from PIL import Image

PALETTE = [
    # Base colours
    "#24273a",
    "#1e2030",
    "#181926",

    # Surfaces
    "#363a4f",
    "#494d64",
    "#5b6078",

    # Overlays
    "#6e738d",
    "#8087a2",
    "#939ab7",

    # Text
    "#a5adcb",
    "#b8c0e0",
    "#cad3f5",

    # Accents
    "#f4dbd6",  # Rosewater
    "#f0c6c6",  # Flamingo
    "#f5bde6",  # Pink
    "#c6a0f6",  # Mauve
    "#ed8796",  # Red
    "#ee99a0",  # Maroon
    "#f5a97f",  # Peach
    "#eed49f",  # Yellow
    "#a6da95",  # Green
    "#8bd5ca",  # Teal
    "#91d7e3",  # Sky
    "#7dc4e4",  # Sapphire
    "#8aadf4",  # Blue
    "#b7bdf8",  # Lavender
]

palette = np.array([
    [int(c[i:i+2], 16) for i in (1, 3, 5)]
    for c in PALETTE
], dtype=np.float32)


@njit(cache=True)
def quantize(pixels, palette, strength=0.65):
    height, width, _ = pixels.shape
    output = np.empty((height, width, 3), dtype=np.uint8)

    work = pixels.copy()

    for y in range(height):
        for x in range(width):
            r = work[y, x, 0]
            g = work[y, x, 1]
            b = work[y, x, 2]

            best_dist = float("inf")
            best_idx = 0

            for i in range(len(palette)):
                dr = r - palette[i, 0]
                dg = g - palette[i, 1]
                db = b - palette[i, 2]

                dist = dr*dr + dg*dg + db*db

                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            nr = palette[best_idx, 0]
            ng = palette[best_idx, 1]
            nb = palette[best_idx, 2]

            output[y, x, 0] = nr
            output[y, x, 1] = ng
            output[y, x, 2] = nb

            er = (r - nr) * strength
            eg = (g - ng) * strength
            eb = (b - nb) * strength

            if x + 1 < width:
                work[y, x+1, 0] += er * 7/16
                work[y, x+1, 1] += eg * 7/16
                work[y, x+1, 2] += eb * 7/16

            if y + 1 < height:
                if x > 0:
                    work[y+1, x-1, 0] += er * 3/16
                    work[y+1, x-1, 1] += eg * 3/16
                    work[y+1, x-1, 2] += eb * 3/16

                work[y+1, x, 0] += er * 5/16
                work[y+1, x, 1] += eg * 5/16
                work[y+1, x, 2] += eb * 5/16

                if x + 1 < width:
                    work[y+1, x+1, 0] += er * 1/16
                    work[y+1, x+1, 1] += eg * 1/16
                    work[y+1, x+1, 2] += eb * 1/16

    return output

def convert(path):
    img = Image.open(path).convert("RGB")
    pixels = np.asarray(img, dtype=np.float32)

    output = quantize(pixels, palette)

    out = Path("output") / (Path(path).stem + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(output).save(out)

    print(f"converter: {path} -> {out}")


def main():
    if len(sys.argv) == 1:
        print("converter: no files specified!")
        sys.exit(1)

    files = []

    for pattern in sys.argv[1:]:
        files.extend(glob.glob(pattern, recursive=True))

    files = sorted(set(files))

    if not files:
        print("converter: no matches found!")
        sys.exit(1)

    for path in files:
        if Path(path).is_file():
            try:
                convert(path)
            except Exception as e:
                print(f"Failed {path}: {e}")


if __name__ == "__main__":
    main()
