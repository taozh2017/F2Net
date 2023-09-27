# Flexible Fusion Network for Multi-modal Brain Tumor Segmentation

> **Authors:**
>
> *Hengyi Yang*,
> [*Tao Zhou*](https://taozh2017.github.io),
> [*Yi Zhou*](https://cse.seu.edu.cn/2021/0303/c23024a362239/page.htm),
> [*Yizhe Zhang*](https://yizhezhang.com/),
> and [*Huazhu Fu*](https://hzfu.github.io/).
>
> 

## 1. Preface

- This repository provides code for "_**Flexible Fusion Network for Multi-modal Brain Tumor Segmentation**_" IEEE JBHI 2023. [Paper](https://ieeexplore.ieee.org/document/10113165/) 


## 2. Overview

### 2.1. Introduction

Automated brain tumor segmentation is crucial for aiding brain disease diagnosis and evaluating disease progress. Currently, magnetic resonance imaging (MRI) is a routinely adopted approach in the field of brain tumor segmentation that can provide different modality images. It is critical to leverage multi-modal images to boost brain tumor segmentation performance. Existing works commonly concentrate on generating a shared representation by fusing multi-modal data, while few methods take into account modality-specific characteristics. Besides, how to efficiently fuse arbitrary numbers of modalities is still a difficult task. In this study, we present a flexible fusion network (termed F2Net) for multi-modal brain tumor segmentation, which can flexibly fuse arbitrary numbers of multi-modal information to explore complementary information while maintaining the specific characteristics of each modality. 
Our F2Net is based on the encoder-decoder structure, which utilizes two Transformer-based feature learning streams and a cross-modal shared learning network to extract individual and shared feature representations. To effectively integrate the knowledge from the multi-modality data, we propose a cross-modal feature-enhanced module (CFM) and a multi-modal collaboration module (MCM), which aims at fusing the multi-modal features into the shared learning network and incorporating the features from encoders into the shared decoder, respectively. Extensive experimental results on multiple benchmark datasets demonstrate the effectiveness of our F2Net over other state-of-the-art segmentation methods.

### 2.2. Framework Overview

<p align="center">
    <img src="imgs/F2Net.png"/> <br />
    <em>
    Figure 1: Overview architecture of our segmentation model, including two key parts, i.e., two Transformer-based specific learning networks and a cross-modal shared learning network.
    </em>
</p>


### 2.3. Qualitative Results

<p align="center">
    <img src="imgs/results.png"/> <br />
    <em> 
    Figure 2: Qualitative Results.
    </em>
</p>

## 3. Proposed Method

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with one NVIDIA RTX 2080Ti GPU.

1. Configuring your environment (Prerequisites):
       
    + Installing necessary packages: `pip install -r requirements.txt`.

2. Downloading necessary data:

    + downloading BraTS 2020 dataset
    which can be found from [Here](https://www.med.upenn.edu/cbica/brats2020/data.html).
   
    + downloading our weights
    which can be found from [Baidu Drive](https://pan.baidu.com/s/1855rwwaaFSM0EJcIlKCtHw) (extraction code: aet5).
    
    + downloading PVTv2 weights and move it into `./networks/pretrained_pth/pvt_v2_b2.pth`,
    which can be found from [GitHub](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth).

3. Preprocessing:

    + Before training, the dataset must be preprocessed, just run `python dateset_preprocess.py`. We provide the train 
    dataset and test dataset in `./data/train_list.txt` and `./data/train_list.txt`.
5. Training:
    
    + After processing training dataset, just run `python train.py --modalities all --model F2Net --batch_size 12` to train our model.
      (you can set your parameters when running the code.)

6. Testing:

    + After you download all the pre-trained model and testing dataset, just run `python test.py --resume ./checkpoint.pth.tar` to generate the final prediction maps.

## 4. Mindspore
You need to run `cd mindspore` first.

1. Environment Configuration:

    + MindSpore: 2.0.0
    + Python: 3.7.0
    + MindCV
  
2. Preprocessing:

    + Before training, the dataset must be preprocessed, just run `python dateset_preprocess.py`. We provide the train 
    dataset and test dataset in `./data/train_list.txt` and `./data/train_list.txt`.

3. Training Configuration:

    + Assigning your costumed path, like `--save_model` and so on in `train.py`.
    
    + Just enjoy it!

4. Testing Configuration:

    + update soon...

## 5. Citation

Please cite our paper if you find the work useful, thanks!

    @ARTICLE{10113165,
       author={Yang, Hengyi and Zhou, Tao and Zhou, Yi and Zhang, Yizhe and Fu, Huazhu},
       journal={IEEE Journal of Biomedical and Health Informatics}, 
       title={Flexible Fusion Network for Multi-modal Brain Tumor Segmentation}, 
       year={2023},
       volume={27},
       number={7},
       pages={3349-3359},
       doi={10.1109/JBHI.2023.3271808}
    }


**[⬆ back to top](#1-preface)**
