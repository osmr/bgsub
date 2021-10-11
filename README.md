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
Usual script run:
```
python subtract_bg.py --service=<service> --token=<your token> --input=<directory with images> --output=<output directory with binary masks>
```
Here:
- `service` is `benzinio` for `benzin.io` service or `removebg` for `remove.bg`,
- `token` is a service API token value, which you will receive after registering on the selected service,
- `input` is a directory with processing JPEG images (can contain subdirectories),
- `output` is a directory with resulted PNG binary masks (it is assumed that all original images had unique names).

Optional parameters:
- `middle` is a directory with intermediate images with original masks obtained from the service (PNG with alpha mask),
- `threshold` is a threshold for mask binarization (default value is 127),
- `url` is a optional custom URL for service.

## Remark

The script does not recalculate the masks if the target images already exist.
