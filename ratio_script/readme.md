This script scans a YOLO-formatted dataset and reports:

- How many images are:
  - positive (at least one bounding box in the label file)
  - negative (label file exists but is empty)
  - missing labels
- Totals per split (`train`, `val`, `test`) and global totals.

It is useful to check the balance between positive and negative images, and to verify that labels exist for all images.

Expected dataset layout (Standard Yolo Format)
	•	<root>/images/train/*.jpg|*.png...
	•	<root>/labels/train/*.txt
	•	<root>/images/val/*.jpg|*.png...
	•	<root>/labels/val/*.txt
	•	(optional) <root>/images/test, <root>/labels/test


From the repository root:

```bash
python ratio_script/ratio.py PATH_TO_YOLO_DATASET
```

Example:

```bash
python ratio_script/ratio.py data/insulator_dataset
```

Typical output looks like:

- Per split:

  - Number of images
  - Positive images (non-empty label files)
  - Negative images (empty label files)
  - Missing labels

- Global totals:

  - Total images
  - Total positives
  - Total negatives
  - Total missing labels

Interpretation