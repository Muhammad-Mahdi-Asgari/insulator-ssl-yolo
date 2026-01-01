Required:

- `input`
  
  Path to a single image file or a directory containing images.
  
- `--weights PATH`

  Path to a YOLO `.pt` weights file. For example:
  - `models/new_model/weights/best.pt` – new high-res model
  - `models/baseline_model/weights/best.pt` – baseline model

- `--output-dir DIR`

  Directory where annotated images (and optional JSON) will be saved. Created if it does not exist.

Common optional arguments:

- `--tile-size INT` (default: `1536`)

  Size of each tile in pixels (square tiles). Increase if your GPU allows it and you want more context per tile; decrease if you get out-of-memory errors.

- `--tile-stride INT` (default: `1024`)

  Step between tile top-left corners. Smaller stride = more overlap, more redundancy, better coverage but more computation. Larger stride = fewer tiles, faster, but higher risk of missing objects at tile boundaries.

- `--conf FLOAT` (default: `0.25`)

  Confidence threshold for detections. Lower values keep more detections (including noisy ones); higher values filter more aggressively.

- `--iou FLOAT` (default: `0.7`)

  IoU threshold used when merging overlapping detections from different tiles. Higher IoU means boxes must overlap more to be merged.

- `--device STR` (default: `"0"`)

  Device string passed to Ultralytics YOLO. Examples:
  - `"0"` – first CUDA GPU
  - `"0,1"` – multi-GPU (if supported)
  - `"cpu"` – CPU only

- `--class-id INT` (default: `0`)

  Class index of the insulator class in the YOLO model. If your model has a single class, this is `0`.

- `--save-json` (flag)

  If set, saves a JSON file per image with all detections (bounding boxes, scores, etc.), useful for analysis or report tables.

- `--max-det INT` (default: `300`)

  Maximum number of detections per image after merging.

- `--line-thickness INT` (default: `2`)

  Thickness of bounding box lines on the output image.

- `--show-labels` (flag)

  Draws class labels on each bounding box.

- `--show-conf` (flag)

  Draws confidence scores alongside labels.

- `--verbose` (flag)

  Prints detailed logs about tiling, merging, and detection counts per tile / image.

### example command:

python image_inference_script/inference_tiled.py \
    /path/to/input_image_or_directory \
    --weights models/new_model/weights/best.pt \
    --output-dir /path/to/output_dir

### example command
python image_inference_script/inference_tiled.py \
    data/eval_stills \
    --weights models/new_model/weights/best.pt \
    --output-dir outputs/new_model_eval \
    --tile-size 1536 \
    --tile-stride 1024 \
    --conf 0.25 \
    --iou 0.7 \
    --show-labels --show-conf