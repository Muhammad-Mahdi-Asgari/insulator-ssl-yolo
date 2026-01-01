import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Tiled inference for YOLO on large images.")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Image or directory of images")
    parser.add_argument("--outdir", type=str, default="runs/tiled_inference", help="Output directory")
    parser.add_argument("--tile-size", type=int, default=1536, help="Tile size (square)")
    parser.add_argument("--overlap", type=float, default=0.3, help="Fractional overlap between tiles [0,1)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="0", help="YOLO device string, e.g. '0' or 'cpu'")
    return parser.parse_args()


def make_tiles(img: np.ndarray, tile_size: int, overlap: float):
    h, w = img.shape[:2]
    stride = int(tile_size * (1.0 - overlap))
    stride = max(1, stride)

    tiles = []
    for y0 in range(0, max(h - tile_size, 0) + 1, stride):
        for x0 in range(0, max(w - tile_size, 0) + 1, stride):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)

            # adjust window to always be tile_size when possible
            y0_adj = max(0, y1 - tile_size)
            x0_adj = max(0, x1 - tile_size)

            crop = img[y0_adj:y1, x0_adj:x1]
            tiles.append((x0_adj, y0_adj, crop))

    # handle tiny images (smaller than tile_size)
    if not tiles:
        tiles.append((0, 0, img.copy()))

    return tiles


def draw_box(canvas, x1, y1, x2, y2, color=(0, 255, 0), label=None):
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    if label is not None:
        cv2.putText(
            canvas,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def run_tiled_inference_on_image(
    model,
    img_path: Path,
    out_path: Path,
    tile_size: int,
    overlap: float,
    conf: float,
    iou: float,
    device: str,
):
    print(f"[INFO] Reading image: {img_path}")
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Failed to read image: {img_path}")
        return

    h, w = img.shape[:2]
    canvas = img.copy()

    tiles = make_tiles(img, tile_size, overlap)
    print(f"[INFO] Generated {len(tiles)} tiles for {img_path.name} (size: {w}x{h})")

    model.to(device)

    total_dets = 0

    for idx, (x0, y0, tile) in enumerate(tiles):
        # YOLO expects RGB
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

        results = model(
            tile_rgb,
            imgsz=tile_size,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        total_dets += len(xyxy)

        for (bx1, by1, bx2, by2), cid, cf in zip(xyxy, cls_ids, confs):
            gx1 = int(bx1 + x0)
            gy1 = int(by1 + y0)
            gx2 = int(bx2 + x0)
            gy2 = int(by2 + y0)

            gx1 = max(0, min(gx1, w - 1))
            gy1 = max(0, min(gy1, h - 1))
            gx2 = max(0, min(gx2, w - 1))
            gy2 = max(0, min(gy2, h - 1))

            label = f"{cid}:{cf:.2f}"
            draw_box(canvas, gx1, gy1, gx2, gy2, color=(0, 255, 0), label=label)

    print(f"[INFO] Total detections for {img_path.name}: {total_dets}")

    # ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        print(f"[ERROR] Failed to write output image: {out_path}")
    else:
        print(f"[INFO] Saved tiled prediction: {out_path}")


def main():
    args = parse_args()
    model_path = Path(args.model)
    source_path = Path(args.source)
    outdir = Path(args.outdir)

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))

    if source_path.is_dir():
        img_paths = [
            p
            for p in sorted(source_path.rglob("*"))
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        ]
    else:
        img_paths = [source_path]

    print(f"[INFO] Source exists: {source_path.exists()} ({source_path})")
    print(f"[INFO] Found {len(img_paths)} image(s).")

    if not img_paths:
        print(f"[WARN] No images found under {source_path}. Check path and extensions.")
        return

    for img_path in img_paths:
        print(f"[INFO] Processing: {img_path}")
        if source_path.is_dir():
            rel = img_path.relative_to(source_path)
        else:
            rel = img_path.name
        out_path = outdir / rel
        run_tiled_inference_on_image(
            model=model,
            img_path=img_path,
            out_path=out_path,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
        )

    print(f"[INFO] Done. Outputs under: {outdir.resolve()}")


if __name__ == "__main__":
    main()
