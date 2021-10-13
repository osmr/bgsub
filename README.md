# Automatic Image Background Subtraction

[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2C3.8-lightgrey.svg)](https://github.com/osmr/bgsub)

This repo contains a script for automatic one-shot image background subtraction task using the appropriate services:
- [benzin.io](https://benzin.io/),
- [remove.bg](https://www.remove.bg/).

## Installation
```
git clone https://github.com/osmr/bgsub.git
cd bgsub
pip install -r requirements.txt
```

## Usage
1. Launch a script for background subtraction via benzin.io/remove.bg service:
```
python subtract_bg_service.py --service=<service> --token=<your token> --input=<directory with images> --output=<output directory with binary masks>
```
Here:
- `service` is `benzinio` for `benzin.io` service or `removebg` for `remove.bg`,
- `token` is a service API token value, which you will receive after registering on the selected service,
- `input` is a directory with processing JPEG images (can contain subdirectories),
- `output` is a directory with resulted PNG binary masks (it is assumed that all original images had unique names).

Optional parameters:
- `middle` is a directory with intermediate images with original masks obtained from the service (PNG with alpha mask),
- `ppdir` is a flag for adding extra parrent+parrent directory to the output one (should use as `--ppdir`).
- `threshold` is a threshold for mask binarization (default value is 127),
- `url` is an optional custom URL for service,
- `jpg` is a flag for forced recompression an input image as JPG (should use as `--jpg`),
- `not-resize` is a flag for suppressing forcible scale the mask to the input image (should use as `--not-resize`).

2. Launch a script for background subtraction via human segmentation network:
```
python subtract_bg_human.py --input=<directory with images> --output=<output directory with binary masks>
```
Here:
- `input` is a directory with processing JPEG images (can contain subdirectories),
- `output` is a directory with resulted PNG binary masks (it is assumed that all original images had unique names).

Optional parameters:
- `ppdir` is a flag for adding extra parrent+parrent directory to the output one (should use as `--ppdir`).
- `use-cuda` is a flag for using CUDA for network inference (should use as `--use-cuda`).

## Remark

The script does not recalculate the masks if the target images already exist.
