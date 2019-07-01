## FeatherNets for [Face Anti-spoofing Attack Detection Challenge@CVPR2019](https://competitions.codalab.org/competitions/20853#results)[1]

## The detail in our paper：[FeatherNets: Convolutional Neural Networks as Light as Feather for Face Anti-spoofing](https://arxiv.org/pdf/1904.09290)

# FeatherNetB Inference Time **1.87ms** In CPU(i7,OpenVINO)

# Params only 0.35M!! FLOPs 80M !! 

# Results on the validation set

|model name | ACER|TPR@FPR=10E-2|TPR@FPR=10E-3|FP|FN|epoch|params|FLOPs|
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|FishNet150| 0.00144|0.999668|0.998330|19|0|27|24.96M|6452.72M|
|FishNet150| 0.00181|1.0|0.9996|24|0|52|24.96M|6452.72M|
|FishNet150| 0.00496|0.998664|0.990648|48|8|16|24.96M|6452.72M|
|MobileNet v2|0.00228|0.9996|0.9993|28|1|5|2.23M|306.17M
|MobileNet v2|0.00387|0.999433|0.997662|49|1|6|2.23M|306.17M
|MobileNet v2|0.00402|0.9996|0.992623|51|1|7|2.23M|306.17M
|MobileLiteNet54|0.00242|1.0|0.99846|32|0|41|0.57M|270.91M|
|MobileLiteNet54-se|0.00242|1.0|0.996994|32|0|69|0.57M|270.91M|
|FeatherNetA|0.00261|1.00|0.961590|19|7|51|0.35M|79.99M|
|FeatherNetB|0.00168|1.0|0.997662|20|1|48|0.35M|83.05M|
|**Ensembled all**|0.0000|1.0|1.0|0|0|-|-|-|

## Recent Update

**2019.4.4**: updata data/fileList.py

**2019.3.10**:code upload for the origanizers to reproduce.

**2019.4.23**:add our paper FeatherNets

# Prerequisites

##  install requeirements
```
conda env create -n env_name -f env.yml
```

## Data


### [CASIA-SURF Dataset](https://arxiv.org/abs/1812.00408)[2]


### Our Private Dataset(Available Soon)



### Data index tree
```
├── data
│   ├── our_realsense
│   ├── Training
│   ├── Val
│   ├── Testing
```
Download and unzip our private Dataset into the ./data directory. Then run data/fileList.py to prepare the file list.

### Data Augmentation

| Method | Settings |
| -----  | -------- |
| Random Flip | True |
| Random Crop | 8% ~ 100% |
| Aspect Ratio| 3/4 ~ 4/3 |
| Random PCA Lighting | 0.1 |


# Train the model

```
#### Train FeatherNetA
```
python main.py --config="cfgs/FeatherNetA-32.yaml" --b 32 --lr 0.01  --every-decay 60 --fl-gamma 3 >> MobileLiteNetA-bs32-train.log
```
#### Train FeatherNetB
```
python main.py --config="cfgs/FeatherNetB-32.yaml" --b 32 --lr 0.01  --every-decay 60 --fl-gamma 3 >> MobileLiteNetB-bs32--train.log

```

>[1] ChaLearn Face Anti-spoofing Attack Detection Challenge@CVPR2019,[link](https://competitions.codalab.org/competitions/20853?secret_key=ff0e7c30-e244-4681-88e4-9eb5b41dd7f7)

>[2] Shifeng Zhang, Xiaobo Wang, Ajian Liu, Chenxu Zhao, Jun Wan, Sergio Escalera, Hailin Shi, Zezheng Wang, Stan Z. Li, " CASIA-SURF: A Dataset and Benchmark for Large-scale Multi-modal Face Anti-spoofing ", arXiv, 2018 [PDF](https://arxiv.org/abs/1812.00408)
