"""
microCT starter: image processing + CT-style stack support.

What it can do right now:
- Demo mode: loads a built-in sample image, segments it, prints metrics, saves figures.
- Stack mode: loads a folder of slices (png/jpg/tif/tiff), builds a 3D volume,
  shows a chosen slice (default: middle), segments it, prints metrics, saves figures.
  Optionally saves a montage of evenly spaced slices.

ct_ready — How this maps to microCT:
  - CT acquisition produces many 2D projection/slice images per rotation.
  - These slices are stacked into a 3D volume (Z, Y, X).
  - Segmentation separates material (bright, high attenuation) from void/air (dark).
  - Metrics like material fraction help characterize structure.
  - Later: registration + difference maps enable in-situ / time-lapse analysis
    (compare same sample before/after loading, heating, etc.).

  How this maps to microCT: CT acquisition produces many 2D slice images
  (one per reconstruction plane or rotation step). Stacking those slices
  into a 3D array (Z, Y, X) gives the volume we analyze slice-by-slice or as a whole.

Run examples:
  python test_image.py --demo
  python test_image.py --stack "path/to/slices_folder"
  python test_image.py --stack "path/to/slices_folder" --slice 50
  python test_image.py --stack "path/to/slices_folder" --montage
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from skimage import io, data
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, binary_opening, disk


def ensure_outdir(outdir: Path) -> None:
    """Create output directory if it does not exist."""
    outdir.mkdir(parents=True, exist_ok=True)


def save_gray(path: Path, img: np.ndarray, title: str) -> None:
    """Save a grayscale image to file."""
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_binary(path: Path, mask: np.ndarray, title: str) -> None:
    """Save a binary mask to file."""
    plt.figure()
    plt.imshow(mask, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def segment_otsu_2d(
    img: np.ndarray,
    *,
    min_obj_px: int = 500,
    min_hole_px: int = 500,
) -> np.ndarray:
    """
    Segment using Otsu threshold + morphological cleanup.
    Returns a boolean mask where True = material (bright), False = void.
    """
    t = threshold_otsu(img)
    mask = img > t

    mask = binary_opening(mask, disk(1))
    mask = remove_small_objects(mask, min_size=min_obj_px)
    mask = remove_small_holes(mask, area_threshold=min_hole_px)
    return mask


def print_basic_metrics(img: np.ndarray, mask: np.ndarray, label: str) -> None:
    """Print intensity and material-fraction metrics for a slice."""
    material_fraction = float(mask.mean())
    air_fraction = 1.0 - material_fraction

    print(f"\n[{label}]")
    print(f"  image shape: {img.shape}")
    print(f"  intensity: min={img.min():.1f}, max={img.max():.1f}, mean={img.mean():.1f}")
    print(f"  material fraction (segmented True): {material_fraction:.4f}")
    print(f"  air/void fraction (1 - material):   {air_fraction:.4f}")


def load_stack(folder: Path) -> np.ndarray:
    """
    Load a folder of 2D slices into a 3D volume with shape (Z, Y, X).
    Supports png, jpg, tif, tiff. RGB images are converted to grayscale.
    Files are sorted by filename so slice order is consistent.
    """
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    # Collect only image files and sort by name for consistent (Z, Y, X) ordering
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

    if not files:
        raise FileNotFoundError(
            f"No image slices found in '{folder}'. "
            f"Expected files with extensions: .png, .jpg, .jpeg, .tif, .tiff"
        )

    slices: list[np.ndarray] = []
    for p in files:
        try:
            img = io.imread(str(p))
        except Exception as e:
            raise RuntimeError(f"Failed to read image '{p}': {e}") from e

        if img.ndim == 3:
            img = img[..., 0]  # Use first channel for RGB
        slices.append(img.astype(np.float32))

    # Stack slices along first axis -> shape (Z, Y, X)
    vol = np.stack(slices, axis=0)

    # Normalize to 0–255 for consistent segmentation
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin) * 255.0

    return vol


def save_montage(vol: np.ndarray, path: Path, n_slices: int = 5, title: str = "Slice montage") -> None:
    """
    Save a montage of n_slices evenly spaced across the volume.
    """
    z_max = vol.shape[0] - 1
    if z_max < 0:
        return

    n = min(n_slices, z_max + 1)
    indices = np.linspace(0, z_max, n, dtype=int)

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.5))
    if n == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        ax.imshow(vol[idx], cmap="gray")
        ax.set_title(f"z={idx}")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_demo(outdir: Path) -> None:
    """Run demo workflow with built-in sample image."""
    ensure_outdir(outdir)

    img = data.camera().astype(np.float32)
    mask = segment_otsu_2d(img)

    save_gray(outdir / "demo_image.png", img, "Demo image (built-in)")
    save_binary(outdir / "demo_segmented.png", mask, "Demo segmentation (Otsu + cleanup)")

    print_basic_metrics(img, mask, "DEMO")


def run_stack(
    folder: Path,
    outdir: Path,
    slice_index: Optional[int] = None,
    save_montage_flag: bool = False,
) -> None:
    """
    Load slice stack, optionally visualize a chosen slice and save montage.
    slice_index: which slice to visualize and segment (default: middle).
    save_montage_flag: if True, save montage of 5 evenly spaced slices.
    """
    ensure_outdir(outdir)

    vol = load_stack(folder)
    n_slices = vol.shape[0]

    # Resolve slice index (default: middle)
    if slice_index is None:
        slice_index = n_slices // 2

    if slice_index < 0 or slice_index >= n_slices:
        raise ValueError(
            f"Slice index {slice_index} out of range [0, {n_slices - 1}]. "
            f"Volume has {n_slices} slices."
        )

    # Extract middle (or chosen) slice and segment it with same method as demo
    mid = vol[slice_index].astype(np.float32)
    mask = segment_otsu_2d(mid)

    # Save middle slice and its segmentation mask to outputs folder
    save_gray(
        outdir / "stack_middle_slice.png",
        mid,
        f"Slice z={slice_index} of {n_slices}",
    )
    save_binary(
        outdir / "stack_middle_segmented.png",
        mask,
        "Segmentation (Otsu + cleanup)",
    )

    # Display the middle slice and segmentation (blocks until user closes windows)
    plt.figure()
    plt.imshow(mid, cmap="gray")
    plt.title(f"Middle slice z={slice_index} of {n_slices}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.imshow(mask, cmap="gray")
    plt.title("Segmented mask (material = white)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    if save_montage_flag:
        save_montage(
            vol,
            outdir / "stack_montage.png",
            n_slices=5,
            title="Stack montage (5 evenly spaced slices)",
        )
        print(f"  saved montage: {outdir / 'stack_montage.png'}")

    # Print volume shape and basic stats (min/max/mean intensity + material fraction)
    print(f"\n[STACK]")
    print(f"  volume shape (Z,Y,X): {vol.shape}")
    print_basic_metrics(mid, mask, f"STACK / SLICE z={slice_index}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="microCT starter: demo + slice-stack processing",
    )
    parser.add_argument("--demo", action="store_true", help="Run built-in demo image workflow")
    parser.add_argument("--stack", type=str, default=None, help="Folder containing slice images")
    parser.add_argument("--out", type=str, default="outputs", help="Output folder for saved figures")
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        metavar="INDEX",
        help="Slice index to visualize and segment (default: middle). Stack mode only.",
    )
    parser.add_argument(
        "--montage",
        action="store_true",
        help="Save montage of 5 evenly spaced slices. Stack mode only.",
    )

    args = parser.parse_args()
    outdir = Path(args.out)

    if args.demo:
        run_demo(outdir)
        print(f"\nSaved outputs to: {outdir.resolve()}")
        return

    if args.stack:
        folder = Path(args.stack)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: '{folder}'. Check the path exists.")
        run_stack(
            folder,
            outdir,
            slice_index=args.slice,
            save_montage_flag=args.montage,
        )
        print(f"\nSaved outputs to: {outdir.resolve()}")
        return

    print("Nothing to do. Try one of these:")
    print("  python test_image.py --demo")
    print('  python test_image.py --stack "path/to/slices_folder"')
    print('  python test_image.py --stack "path/to/slices_folder" --slice 50')
    print('  python test_image.py --stack "path/to/slices_folder" --montage')


if __name__ == "__main__":
    main()
