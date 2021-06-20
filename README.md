## Hair Segmentation 
해당 레포지토리는 인공지능 온라인 경진대회 - 한국인 헤어스타일 세그멘테이션 대회 준비를 위한 깃허브입니다. 총 2가지의 데이터셋에 대해서 각각 pretrained 모델을 만들어두었습니다. 사용된 모델들의 경우 https://github.com/qubvel/segmentation_models.pytorch 에 있는 주요 SOTA 모델들을 사용했습니다. 
- Encoder : efficientnet, se_resnext
- Decoder : Unet, FPN, LinkNet, DeepLabv3+, Unet++

※ 참고 : 대회 규정상 **헤어스타일 변환 경진대회**으로 학습된 모델은 사용하면 안된다고 합니다. Google Drive에도 해당 가중치는 모두 제거해두었습니다. 

## Prerequisites
`pip install -r requirements.txt` 

## Dataset 
- [헤어스타일 변환 경진대회 (Hairstyle Translation AI Competetion)](https://github.com/niahair/ganhackerton) (contain 250,000 images and masks hair segmentation) 
- [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) (contain 29.300 images and masks hair segmentation)

**헤어스타일 변환 경진대회**에서 제공된 한국인의 25만장의 이미지와 레이블을 통해 만든 Hair-Segmentation 모델입니다. 데이터의 경우 아래의 명령어를 통해서 `data` 폴더에 저장해줘야 합니다. 

`git clone https://github.com/niahair/ganhackerton.git`

**CelebAMask-HQ** 데이터의 경우 [링크](https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv)에서 다운받아줄 수 있습니다. 2.9G의 데이터를 다운받아주고 `data` 폴더 내에 저장 해주면 됩니다. 그리고 preprocess-CelebAMask-HQ.ipynb를 실행시켜줍니다. 

두 개의 파일에 대한 이미지를 보시려면 visualization.ipynb을 확인해주시기 바랍니다. 

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


## Train model 

```
python train_korean.py [--img_size IMG_SIZE] [--num_workers NUM_WORKERS] [encoder_type ENCODER_TYPE] [] ...

python train_celeb.py [--img_size IMG_SIZE] [--num_workers NUM_WORKERS] [encoder_type ENCODER_TYPE] [] ...

optional arguments:
    --img_size:           size image input, default 512
    --num_workers:        num workers of dataloader, default 0
    --encoder_type:       backbone model, default 'timm-efficientnet-b0'
    --decoder_type:       decoder model, default 'Unet'
    --scheduler:          scheduler type, default 'GradualWarmupSchedulerV2'
    --encoder_lr:         learning rate of encoder, default 3e-5
    --min_lr:             minimum learning rate, default 1e-6
    --batch_size:         batch size training, default 4
    --weight_decay:       weight decay, default 1e-6
    --apex:               use apex, default False   
```

학습에 사용한 pretrained 모델의 가중치에 대해서는 [Google-Drive](https://drive.google.com/drive/folders/19jm8wjBH6Pf3XJBXfPJ_-CLi5W76fszx?usp=sharing)에서 다운받을 수 있습니다. 파일명 뒤에 이름에 따라서 아래의 3가지 형태로 나뉘게됩니다. 
- light : ganhackerton을 가볍게 학습한 가중치
- heavy : ganhackerton을 무겁게 학습한 가중치
- celeb : CelebAMask를 학습한 가중치 

## Inference model 

```
python inference.py [--img_size IMG_SIZE] [--num NUM] 

optional arguments:
    --img_size:           size image input, default 512
    --num:                demo image number, default 1
```


## Result  

![](https://drive.google.com/uc?export=view&id=18y9s2TZBXI6SlN_d0d3vks8tmnkmC8Mj)

|   	| Encoder        	| Decoder    	| Dice   	| Data   	|
|---	|----------------	|------------	|--------	|--------	|
| 1 	| EfficientNetB0 	| Unet       	| 0.9644 	| Korean 	|
| 2 	| EfficientNetB0 	| FPN        	| 0.9626 	| Korean 	|
| 3 	| EfficientNetB0 	| LinkNet    	| 0.9597 	| Korean 	|
| 4 	| EfficientNetB0 	| DeepLabv3+ 	| 0.9625 	| Korean 	|
| 5 	| EfficientNetB0 	| Unet       	| 0.9657 	| Celeb  	|
| 6 	| EfficientNetB0 	| FPN        	| 0.0000 	| Celeb  	|
| 7 	| EfficientNetB0 	| LinkNet    	| 0.0000 	| Celeb  	|
| 8 	| EfficientNetB0 	| DeepLabv3+ 	| 0.0000 	| Celeb  	|

다른 모델들에 대한 실험 결과는 [스프레드시트](https://docs.google.com/spreadsheets/d/1-TDV4K2PAI0DBcMOHyV4d1TNtKjO0ORMourQx48rvgA/edit?usp=sharing)에서 확인이 가능합니다.  

## Reference 
```
@inproceedings{CelebAMask-HQ,
  title={MaskGAN: Towards Diverse and Interactive Facial Image Manipulation},
  author={Lee, Cheng-Han and Liu, Ziwei and Wu, Lingyun and Luo, Ping},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```
