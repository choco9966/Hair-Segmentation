## Dataset 
- [헤어스타일 변환 경진대회 (Hairstyle Translation AI Competetion)](https://github.com/niahair/ganhackerton) (contain 250,000 images and masks hair segmentation) 

**헤어스타일 변환 경진대회**에서 제공된 한국인의 25만장의 이미지와 레이블을 통해 만든 Hair-Segmentation 모델입니다. 데이터의 경우 아래의 명령어를 통해서 `data` 폴더에 저장해줘야 합니다. 

`git clone https://github.com/niahair/ganhackerton.git`

**CelebAMask-HQ** 데이터의 경우 [링크](https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv)에서 다운받아줄 수 있습니다. 2.9G의 데이터를 다운받아주고 `data` 폴더 내에 저장 해주면 됩니다. 그리고 preprocess-CelebAMask-HQ.ipynb를 실행시켜줍니다. 

**Data structure**
```
├── data 
│   ├── ganhackerton
│   │   ├──dataset
│   │   │   ├── partition1
│   │   │   ├── partition2
│   │   │   ├── partition3
...
│   ├── CelebAMask-HQ
│   │   ├──image
│   │   │   ├── 1.jpg
│   │   │   ├── 2.jpg
│   │   │   ├── 3.jpg
...
│   │   ├──mask
│   │   │   ├── 1.jpg
│   │   │   ├── 2.jpg
│   │   │   ├── 3.jpg
...

```



## Reference 
```
@inproceedings{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
