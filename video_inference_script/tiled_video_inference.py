import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ---------- TILING HELPERS ----------

def generate_tiles(frame, tile_size=896, overlap=0.3):
    """
    Split a frame into overlapping tiles.
    Returns:
      tiles: list of (tile_img, x0, y0)
    """
    h, w, _ = frame.shape
    stride = int(tile_size * (1 - overlap))
    stride = max(1, stride)

    tiles = []

    if h <= tile_size and w <= tile_size:
        tiles.append((frame.copy(), 0, 0))
        return tiles

    y = 0
    while y < h:
        x = 0
        y_end = min(y + tile_size, h)
        y_start = max(0, y_end - tile_size)

        while x < w:
            x_end = min(x + tile_size, w)
            x_start = max(0, x_end - tile_size)

            tile = frame[y_start:y_end, x_start:x_end]
            # If tile is smaller than tile_size (at borders), pad to square
            th, tw, _ = tile.shape
            if th != tile_size or tw != tile_size:
                pad = np.zeros((tile_size, tile_size, 3), dtype=frame.dtype)
                pad[0:th, 0:tw] = tile
                tile = pad

            tiles.append((tile, x_start, y_start))
            x += stride
        y += stride

    return tiles


def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    box1: [N, 4], box2: [M, 4], format xyxy
    """
    # Intersection
    tl = torch.max(box1[:, None, :2], box2[:, :2])   # [N,M,2]
    br = torch.min(box1[:, None, 2:], box2[:, 2:])   # [N,M,2]
    wh = (br - tl).clamp(min=0)                      # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]                # [N,M]

    # Areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


def nms(boxes, scores, iou_thr=0.5):
    """
    Simple NMS in PyTorch.
    boxes: [N,4] (xyxy), scores: [N]
    returns: indices to keep
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    idxs = scores.argsort(descending=True)
    keep = []

    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious <= iou_thr]

    return torch.tensor(keep, dtype=torch.long)


def run_tiled_yolo_on_frame(frame, model, tile_size=896, overlap=0.3,
                            conf_thres=0.2, iou_merge=0.5, device="cuda"):
    """
    Run YOLO on a single frame using tiling, then merge detections.
    Returns:
      frame_with_boxes, detections
      detections: list of dicts {box, conf, cls}
    """
    h, w, _ = frame.shape
    tiles = generate_tiles(frame, tile_size=tile_size, overlap=overlap)

    all_boxes = []
    all_scores = []
    all_classes = []

    for tile_img, x_off, y_off in tiles:
        # BGR -> RGB for YOLO
        tile_rgb = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)

        results = model(
            tile_rgb,
            conf=conf_thres,
            iou=0.7,
            verbose=False,
            device=device
        )

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()   # [N,4]
        scores = r.boxes.conf.cpu().numpy()  # [N]
        classes = r.boxes.cls.cpu().numpy()  # [N]

        # Map tile boxes back to full-frame coords
        for b, s, c in zip(boxes, scores, classes):
            x1, y1, x2, y2 = b
            x1 += x_off
            x2 += x_off
            y1 += y_off
            y2 += y_off

            # Clip to frame just in case
            x1 = np.clip(x1, 0, w - 1)
            x2 = np.clip(x2, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            y2 = np.clip(y2, 0, h - 1)

            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(s)
            all_classes.append(int(c))

    if not all_boxes:
        return frame, []

    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    classes_t = torch.tensor(all_classes, dtype=torch.long)

    # Single-class case: global NMS
    keep = nms(boxes_t, scores_t, iou_thr=iou_merge)

    boxes_t = boxes_t[keep]
    scores_t = scores_t[keep]
    classes_t = classes_t[keep]

    detections = []
    out = frame.copy()

    for box, score, cls in zip(boxes_t, scores_t, classes_t):
        x1, y1, x2, y2 = box.int().tolist()
        conf = float(score.item())

        # draw rectangle
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"insulator {conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        detections.append({
            "box": [x1, y1, x2, y2],
            "conf": conf,
            "cls": int(cls.item()),
        })

    return out, detections


# ---------- MAIN VIDEO LOOP ----------

def main():
    parser = argparse.ArgumentParser(description="Tiled YOLO video inference")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLO weights (.pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video path or camera index (e.g. 0)"
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=896,
        help="Tile size (pixels)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.3,
        help="Tile overlap fraction (0–1)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--iou-merge",
        type=float,
        default=0.5,
        help="IoU threshold for merging boxes across tiles"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="Optional output video path (e.g. runs/video_out.mp4)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live window"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Video source (int for webcam)
    try:
        src_int = int(args.source)
        cap = cv2.VideoCapture(src_int)
    except ValueError:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    # Prepare writer if saving
    writer = None
    if args.save_path:
        out_path = Path(args.save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        print(f"Saving output video to: {out_path}")

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t_start = time.time()

        out_frame, dets = run_tiled_yolo_on_frame(
            frame,
            model,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf_thres=args.conf,
            iou_merge=args.iou_merge,
            device=args.device,
        )

        t_end = time.time()
        fps_inst = 1.0 / max(1e-6, (t_end - t_start))

        # Overlay FPS
        cv2.putText(
            out_frame,
            f"FPS: {fps_inst:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if args.show:
            cv2.imshow("Tiled YOLO video", out_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer is not None:
            writer.write(out_frame)

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    total_time = time.time() - t0
    print(f"Processed {frame_idx} frames in {total_time:.1f}s")


if __name__ == "__main__":
    main()
