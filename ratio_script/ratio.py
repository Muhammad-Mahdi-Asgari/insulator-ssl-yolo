#!/usr/bin/env python3
import sys
from pathlib import Path

SPLITS = ["train", "val", "test"]

def count_split(img_dir, lbl_dir):
    images = sorted([p for p in img_dir.iterdir() if p.is_file()])
    total = len(images)
    pos = 0
    neg = 0
    missing = 0

    for img in images:
        stem = img.stem
        lbl = lbl_dir / f"{stem}.txt"
        if not lbl.exists():
            missing += 1
            continue

        txt = lbl.read_text().strip()
        if txt == "":
            neg += 1
        else:
            pos += 1

    return total, pos, neg, missing


def main():
    if len(sys.argv) != 2:
        print("Usage: python count_pos_neg_ratio.py <data_root>")
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()

    total_all = pos_all = neg_all = miss_all = 0

    for split in SPLITS:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"[WARN] {split} split missing, skipping.")
            continue

        total, pos, neg, missing = count_split(img_dir, lbl_dir)

        total_all += total
        pos_all += pos
        neg_all += neg
        miss_all += missing

        print(f"\n=== {split.upper()} ===")
        print(f"Images        : {total}")
        print(f"  Positives   : {pos}")
        print(f"  Negatives   : {neg}")
        print(f"  Missing lbl : {missing}")

    print("\n=== GLOBAL TOTAL ===")
    print(f"Images        : {total_all}")
    print(f"  Positives   : {pos_all}")
    print(f"  Negatives   : {neg_all}")
    print(f"  Missing lbl : {miss_all}")

    if total_all > 0:
        print(f"\nPositive ratio: {pos_all/total_all:.3f}")
        print(f"Negative ratio: {neg_all/total_all:.3f}")


if __name__ == "__main__":
    main()