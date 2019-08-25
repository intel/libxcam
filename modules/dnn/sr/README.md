## REQUIREMENTS

This is written in Python 3.6.
The following packages are needed -
```
numpy
torch
torchvision
cv2
Pillow
``` 

## USAGE

### Pretrained models

1. Use this [link](https://drive.google.com/drive/folders/1ijaifz2YxUziiQZL8lrpo8urUTZc0SPq?usp=sharing) to download all the pretrained models.
2. Put the contents of each folder, named by the model, in `architecture/{model}/pretrained_models/` 

### Super-resolving images or videos

1. Configure `settings`, in `main.py`, for `input` and `output` folders.
2. Put all the images and videos to be SR'ed in `settings['input']`
3. Choose the model to be used
4. Run `main.py`
5. Find all the upscaled media in `settings['output']`

### Benchmarking

1. Configure `settings` in `test_quality.py` for `test_dataset_folder` and `output_dataset_folder`
2. Download this benchmarking [dataset](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) and extract its contents in `setting['test_dataset_folder']`.
3. Change other settings if need be.
4. Run `test_quality.py` 

### My Hardware Specs
```
Intel Core i7-7700HQ CPU @ 2.80 GHZ
NVIDIA GeForce GTX 1050 Ti (4GB)
8GB DDR4 RAM
```