"""
visualize.py  –  LBM Frame Visualization
SE3082 Parallel Computing – Assignment 03

Makes animated GIFs from PGM or PPM frames, and generates a side-by-side
comparison snapshot for the report.

Usage:
    python visualize.py --gif serial          # GIF from frames/serial/ (grayscale)
    python visualize.py --gif serial_ppm      # GIF from frames/serial_ppm/ (color vorticity)
    python visualize.py --gif openmp_ppm      # OpenMP vorticity GIF
    python visualize.py --compare 5000        # side-by-side snapshot at step 5000
    python visualize.py --diff serial openmp  # pixel-diff correctness check
"""

import argparse, re, sys
from pathlib import Path
from PIL import Image
import numpy as np

FRAMES_ROOT = Path("frames")
ANIM_DIR    = Path("animations")
PLOTS_DIR   = Path("plots")

# ── helpers ──────────────────────────────────────────────────────────────────
def frame_path(folder, step):
    ext = "ppm" if "ppm" in folder else "pgm"
    return FRAMES_ROOT / folder / f"frame_{step:05d}.{ext}"


def load_frames(folder, step=None):
    """Load all (or one specific) frame(s) from a folder, sorted by step."""
    d = FRAMES_ROOT / folder
    if not d.exists():
        sys.exit(f"Folder not found: {d}")

    if step is not None:
        ext = "ppm" if "ppm" in folder else "pgm"
        p = d / f"frame_{step:05d}.{ext}"
        if not p.exists():
            sys.exit(f"Frame not found: {p}")
        return [Image.open(p)]

    files = sorted(d.glob("frame_*.p?m"))
    if not files:
        sys.exit(f"No frames found in {d}")
    return [Image.open(f) for f in files]


# ── commands ─────────────────────────────────────────────────────────────────
def make_gif(folder, delay_ms=80):
    """Create an animated GIF from all frames in a folder."""
    ANIM_DIR.mkdir(exist_ok=True)
    print(f"Loading frames from frames/{folder}/ …")
    frames = load_frames(folder)
    # Convert PPM (RGB) to P (palette) for GIF compatibility
    frames_p = [f.convert("P", palette=Image.ADAPTIVE, colors=256)
                for f in frames]

    out = ANIM_DIR / f"{folder}.gif"
    frames_p[0].save(
        out,
        save_all=True,
        append_images=frames_p[1:],
        duration=delay_ms,
        loop=0,
    )
    print(f"  Saved → {out}  ({len(frames)} frames, {delay_ms} ms/frame)")


def make_snapshot(folder_a, folder_b, step):
    """Side-by-side PNG of two folders at a given step."""
    PLOTS_DIR.mkdir(exist_ok=True)
    img_a = load_frames(folder_a, step)[0]
    img_b = load_frames(folder_b, step)[0]

    # Paste side by side with a 4-px separator
    w, h  = img_a.size
    sep   = 4
    total = Image.new("RGB", (w * 2 + sep, h), (200, 200, 200))
    total.paste(img_a.convert("RGB"), (0, 0))
    total.paste(img_b.convert("RGB"), (w + sep, 0))

    out = PLOTS_DIR / f"compare_{folder_a}_vs_{folder_b}_step{step:05d}.png"
    total.save(out)
    print(f"  Saved → {out}")


def diff_frames(folder_a, folder_b):
    """Pixel-level correctness check between two implementations."""
    d_a = FRAMES_ROOT / folder_a
    d_b = FRAMES_ROOT / folder_b
    ext = "ppm" if "ppm" in folder_a else "pgm"

    files_a = sorted(d_a.glob(f"frame_*.{ext}"))
    if not files_a:
        sys.exit(f"No frames in {d_a}")

    print(f"Comparing {folder_a} vs {folder_b} …")
    max_overall = 0
    for fa in files_a:
        fb = d_b / fa.name
        if not fb.exists():
            continue
        arr_a = np.array(Image.open(fa))
        arr_b = np.array(Image.open(fb))
        diff  = np.abs(arr_a.astype(int) - arr_b.astype(int))
        md    = int(diff.max())
        if md > max_overall:
            max_overall = md
        if md > 0:
            print(f"  {fa.name}  max_diff={md}  mean_diff={diff.mean():.3f}")

    if max_overall == 0:
        print("  All frames are pixel-identical. ✓")
    else:
        print(f"  Max pixel difference across all frames: {max_overall}")
        if max_overall <= 2:
            print("  Within acceptable floating-point tolerance. ✓")
        else:
            print("  Large differences — check implementation correctness.")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="LBM frame visualizer — GIF, snapshots, correctness diffs"
    )
    ap.add_argument("--gif", metavar="FOLDER",
                    help="Make animated GIF from frames/<FOLDER>/")
    ap.add_argument("--delay", type=int, default=80,
                    help="Milliseconds per frame in GIF (default 80 = ~12 fps)")
    ap.add_argument("--compare", metavar="STEP", type=int,
                    help="Side-by-side snapshot at this step number")
    ap.add_argument("--folders", nargs=2, metavar=("A", "B"),
                    help="Two folders to compare (used with --compare and --diff)")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"),
                    help="Pixel-level correctness check between two folders")
    args = ap.parse_args()

    if not any([args.gif, args.compare, args.diff]):
        ap.print_help()
        return

    if args.gif:
        make_gif(args.gif, delay_ms=args.delay)

    if args.compare is not None:
        folders = args.folders or ["serial_ppm", "openmp_ppm"]
        make_snapshot(folders[0], folders[1], args.compare)

    if args.diff:
        diff_frames(args.diff[0], args.diff[1])


if __name__ == "__main__":
    main()
