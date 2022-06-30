# Feature Erasing and Diffusion Network for Occluded Person Re-Identification (CVPR2022)
Pytorch implementation for the occluded person reid algorithm described in the paper[Feature Erasing and Diffusion Network for Occluded Person Re-Identification (CVPR2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Feature_Erasing_and_Diffusion_Network_for_Occluded_Person_Re-Identification_CVPR_2022_paper.pdf)

## Pipline
<div align="center">
  <img src="Figures/Structure.png">
 </div>
 
 ## Experiment Results on Holistic and Occluded Person ReID Datasets
 <div align="center">
  <img src="Figures/Result1.png" width="400px"/>
 </div>
 
 <div align="center">
  <img src="Figures/Result2.png" width="400px"/>
 </div>
 
 ## Retrieve Comparison between [TransReID](https://github.com/damo-cv/TransReID) 
 <div align="center">
  <img src="Figures/Comparison.png" width="700px"/>
 </div>
 
 
 ## Requirements
 ### Installation
 Please refer to [TransReID](https://github.com/damo-cv/TransReID) 
 ### Dataset Preparation
 Please download Occluded-Duke dataset and [cropped patches](https://drive.google.com/file/d/1lYTBokHR8pkbjz_LPhZjh4ij8B-FI1LA/view?usp=sharing). Meanwhile place cropped patches into Occluded-Duke (just because of dataloader).
 ### Pretrained Model Preparison
 Please download pretrained [ViT backbone](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) in advance.
 
 ## Model training and testing
 before training and testing, please update config file accordingly.  Around 13G GPU memory is required. 
 ~~~~~~~~~~~~~~~~~~
   python train.py 
 ~~~~~~~~~~~~~~~~~~

## Citation

If you find this code useful for your research, please cite our paper

```
@inproceedings{wang2022feature,
  title={Feature Erasing and Diffusion Network for Occluded Person Re-Identification},
  author={Wang, Zhikang and Zhu, Feng and Tang, Shixiang and Zhao, Rui and He, Lihuo and Song, Jiangning},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4754--4763},
  year={2022}
}
```

## Contact

If you have any question, please feel free to contact us. E-mail: [zhikang.wang@monash.edu](zhikang.wang@monash.edu) 
