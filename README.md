# SpaceshipNet
Code of Data-Free Knowledge Distillation via Feature Exchange and Activation Region Constraint

~~The code is currently being prepared, and we will release it as soon as possible.~~



During the preparation process before the paper submission, the code has been enhanced multiple times, and there are many redundant codes. The currently released code has been simply modified and refactored. We also conducted a simple test on CIFAR-100. When the teacher network is ResNet-34 and the student network is ResNet-18, we conducted experiments using 6 different random seeds (0, 20, 40, 60, 80, 100). The best result among them was 77.43%, the average result was 77.15%, the standard deviation was 0.21%, and the result reported in our paper is 77.41%.




We will continue optimize the code to fix any issues reported.



## Getting Started



## Prerequisites

- Linux or macOS

- Python 3 (We used python 3.8)

- Pytorch (Our pytorch version is '2.0.1+cu117')

  We tested the code using Python 3.8 with PyTorch (version: 2.0.1+cu117) on a Tesla A100, and also tested it using Python 3.7 with PyTorch (version: 1.7.1+cu110) on a GeForce RTX 3090.





### Datasets

---

- CIFAR-100

  Place the 'cifar-100-python' folder in your dataset path, e.g., ../../dataset, the cifar-100-python folder can be downloaded by using

  ```python
  import torchvision
  trainset = torchvision.datasets.CIFAR10(root='SpaceshipNet-main/dataset', train=True,
                                          download=True)
  ```

  

- CIFAR-10

  Place the 'cifar-10-batches-py' folder in your datasets path, e.g., ../../dataset, the cifar-100-python folder can be downloaded by using

  ```python
  import torchvision
  trainset = torchvision.datasets.CIFAR100(root='SpaceshipNet-main/dataset', train=True,
                                          download=True)
  ```





### Pretrained Teachers

---

The pre-trained teacher networks of ResNet-34, WRN-40-2, and VGG-11 for CIFAR-10 and CIFAR100 are downloaded from [https://www.dropbox.com/sh/w8xehuk7debnka3/AABhoazFReE_5mMeyvb4iUWoa?dl=0](https://www.dropbox.com/sh/w8xehuk7debnka3/AABhoazFReE_5mMeyvb4iUWoa?dl=0) provided by the code of [1].

Place the downloaded teacher weights in 'code\SpaceshipNet\pretrained\MosaicKD_state_dict\cifar10' and 'code\SpaceshipNet\pretrained\MosaicKD_state_dict\cifar100'. 

For your convenience, I have pack the pretrained teacher weights and uploaded it to Google Drive. The link can be found here: 

[https://drive.google.com/file/d/1kpqggLkxUrF8uXrnkhtFCJBxQNHN-uSA/view?usp=sharing](https://drive.google.com/file/d/1kpqggLkxUrF8uXrnkhtFCJBxQNHN-uSA/view?usp=sharing). Please unzip the pretrained folder and place it at 'code\SpaceshipNet\pretrained'.




### Run SpaceshipNet

---

- CIFAR-100 ResNet-34 - ResNet-18

  ```
  cd code/SpaceshipNet 
  python train/train_cvpr2023.py --config scripts/DFKD_Cifar100_ResNet34_ResNet18_V37_conv012_seed40.yaml
  ```

  





## References

[1] Gongfan Fang, Yifan Bao, Jie Song, Xinchao Wang, Donglin Xie, Chengchao Shen, and Mingli Song. Mosaicking to distill: Knowledge distillation from out-of-domain data. NeurIPS, 34, 2021.



