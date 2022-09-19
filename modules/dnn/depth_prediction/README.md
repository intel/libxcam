This is the a depth estimation models using the method described in

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)
>
> [Github](https://github.com/nianticlabs/monodepth2)
>
<p align="center">
 <img src="assets/clip.jpg" alt="example input output image" width="600" />
</p>



We have implemented the deployment of this model on openvino and can inference on hardware accelerators of various Intel platforms.

## Python
### Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
pip install openvino-dev==2022.1.0
conda install opencv=3.3.1               
```


### Prediction for a single image  

You can download our pre trained and converted [IR format model](https://drive.google.com/drive/folders/19MGkSv1TmkzLCIUJT1Sywd9gneDhUKm9?usp=sharing) and put it into the `models__cvt/` folder.

Modify 'image_path, model_xml, model_bin, output_directory' in the file 'inference.py' to the appropriate path.

Then you can predict scaled disparity for a single image with:

```shell
python inference.py
```

You can also visit [monodepth2](https://github.com/nianticlabs/monodepth2) and train the model with your own dataset. 

After the training, you can download the [folder](https://drive.google.com/drive/folders/1nIHY-36OuQEDiKxcZSdDpu_iuGUCAlJA?usp=sharing) and convert your model into .onnx format file with:

```shell
python model_converter.py --model_name your_model_name
```

Then use openvino's tool library 'mo_onnx.py' script converts the .onnx model into IR models of .xml and .bin.

## C++
### Setup
Configure openvino_2022.1.0.643 and opencv_3.3.1 .

### Prediction for a single image

You can download our pre trained and converted [IR format model](https://drive.google.com/drive/folders/19MGkSv1TmkzLCIUJT1Sywd9gneDhUKm9?usp=sharing) and put it into the `models_cvt/` folder.

Then you can predict scaled disparity for a single image by running 'inference.cpp'
