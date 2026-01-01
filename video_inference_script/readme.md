Run tiled YOLO detection on each frame of a video and then apply object tracking (e.g. BoT-SORT) to get smoother trajectories and more stable IDs across frames.

Pipeline per frame:

1. Read frame from video.
2. Optionally resize frame.
3. Split into overlapping tiles.
4. Run YOLO detection per tile.
5. Merge detections in global coordinates using NMS.
6. Pass detections to the tracker.
7. Draw tracked boxes and write frame to output video.

This is the script to use for:

- Evaluating performance on real flight footage.
- Producing demo videos for the report.

Basic usage

Required:

- `input`

  Path to an input video file (e.g. `.mp4`, `.avi`, etc.).

- `--weights PATH`

  Path to YOLO `.pt` weights (baseline or new model).

- `--output PATH`

  Path for the output video file with detections drawn.

Common optional arguments:

- `--tile-size INT` (default: `1536`)

  Tile size in pixels. Same role as in `inference_tiled.py`.

- `--tile-stride INT` (default: `1024`)

  Stride between tiles. Same idea as for still images: smaller stride = more overlap and more computation.

- `--conf FLOAT` (default: `0.25`)

  Detection confidence threshold before tracking.

- `--iou FLOAT` (default: `0.7`)

  IoU threshold used when merging detections across tiles and for NMS.

- `--device STR` (default: `"0"`)

  Device string for YOLO (e.g. `"0"` or `"cpu"`).

- `--class-id INT` (default: `0`)

  Class index of the insulator class.

- `--tracker STR` (default: `"botsort.yaml"`)

  Tracker configuration file used by Ultralytics. Typically left as default unless you want to tweak tracking.

- `--max-det INT` (default: `300`)

  Maximum detections per frame.

- `--line-thickness INT` (default: `2`)

  Bounding box line thickness.

- `--show-labels` (flag)

  Draw class labels on the video.

- `--show-conf` (flag)

  Draw confidence scores.

- `--resize-long INT` (optional)

  If set, resizes the long side of the frame to this many pixels before tiling (keeps aspect ratio). Useful if your raw video is extremely large.

- `--fps-limit FLOAT` (optional)

  Optional limit on the output FPS. If set, the script can skip frames or adjust timing so that the output video does not explode in size.

- `--verbose` (flag)

  Prints more detailed logging per frame and per tile.

Example command:

python video_inference_script/tiled_video_inference.py \
    /path/to/input_video.mp4 \
    --weights models/new_model/weights/best.pt \
    --output /path/to/output_video.mp4
