# Ko-UniVL

## 한국어 비디오 캡셔닝 모델
본 모델은 Microsoft의 [UniVL](https://github.com/microsoft/UniVL) 모델을 사용해 개발된 한국어 비디오 캡셔닝 모델입니다.  
비디오 캡셔닝 분야의 모델들과 데이터셋들이 모두 영어를 기준으로 제작되었다는 점에서 한국어만을 위한 비디오 캡셔닝 모델을 개발하고자 하였습니다.  
UniVL 모델의 Text Encoder를 BERT 모델에서 koBERT 모델로 교체하는 방식으로 모델을 수정하였습니다.  
한국어 Input 한국어 Output을 위해 UniVL 개발진이 모델 학습에 사용한 데이터(HowTo100M(일부), YouCook2, MSR-VTT)의 캡션을 한국어로 번역하여 학습을 진행하였습니다.  
HowTo100M 데이터셋의 경우 원본 데이터가 약 1억3천만개의 비디오 영상으로 구성되어있습니다. 그러나 제한된 학습 환경으로 인해 일부(약 1만개) 데이터를 다운로드 후 전처리하여 번역해 사용했고 한국어 번역판 데이터와 원본 데이터의 수 차이가 막대해 한국어 번역판의 성능은 좋지 않은 편입니다.  
1억 3천만개의 동영상을 다루는 HowTo100M 데이터셋으로 사전 학습한 파일을 UniVL GitHub에서 제공하고 있으며, 파인 튜닝 시 해당 파일을 적용하면 더 좋은 성능을 얻을 수 있습니다. 
*본 Repo는 Caption Task만을 다룹니다*


## 한국어 모델 전환 과정
### Korean HowTo100M dataset 구성
HowTo100M dataset을 구성하는 영상들의 video id를 사용해 [Pytube](https://pytube.io/en/latest/) 라이브러리로 약 1만여개의 동영상을 직접 다운로드하였습니다.  
이후 동영상들의 S3D feature를 추출하여 캡션들을 한국어로 번역하는 과정을 거쳤습니다.   


### koBERT
기존에 UniVL에서 Text Encoder 모델로써 사용되던 bert 모델을 koBERT 모델로 교체했습니다.  
사용한 koBERT 모델은 [SKT-Brain koBERT](https://github.com/SKTBrain/KoBERT)이며, 해당 repo에서 config.json 파일, model binary 파일, vocab 파일을 다운로드 해 적용했습니다.
> UniVL 모델은 한국어를 지원하는 Bert 계열의 모델 중 하나인 Bert base multilingual model의 적용이 가능하도록 설계되어있습니다.  
그러나 한국어 토크나이징이 전혀 이루어지지 않는 Issue로 인해 multilingual 모델 코드 상의 수정 및 재학습이 이루어져야하는 상태입니다.  
`https://github.com/google-research/bert/pull/228/files`  
KoBERT 모델 개발의 목적이 Bert multilingual 모델의 낮은 한국어 성능을 개선하기 위함이었다는 점과 여건 상 위 과정을 거치기 어렵다는 이유로 한국어 특성을 잘 반영하는 KoBERT 모델로의 교체가 적절하다 판단했습니다. 
### Tokenizer
영어에 적합하게 설계되어있던 BertTokenizer를 KoBertTokenizer로 교체했습니다.  
사용한 Tokenizer 모델은 [KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)이며 해당 레포에서 필요한 폴더를 다운로드해 적용했습니다.  
BertTokenizer가 사용되는 부분을 KoBertTokenizer로 수정하고 KoBertTokenizer와 기존 UniVL 코드 연결을 위해 dataloader 등 다른 파일들을 수정했습니다.



## 학습 과정
<img src="imgs/training process.JPG">
생성된 캡션들은 "KoUniVL/ckpts/ckpt_dataname_caption" 경로의 `hyp.txt` 파일로 확인할 수 있습니다.  
생성된 캡션들에 대한 정답 문장은 동일한 경로의 `ref.txt` 파일에서 확인할 수 있습니다. 


## 학습 환경
`*Window 환경에서의 충돌이 존재할 수 있습니다. Rinux 환경을 권장합니다.*`
- Rinux(Ubuntu) 
- Tesla V100
- CUDA 11.4



## Requirements
- `python==3.6.9`
- `torch==1.10.2`
- `tqdm`
- `boto3`
- `requests`
- `pandas`
- `pickle5`
- `nlg-eval (Install Java 1.8.0 (or higher) firstly)`
```
conda create -n py_univl python=3.6.9 tqdm boto3 requests pandas pickle5
conda activate py_univl
pip install torch==1.10.2
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```
#### KoBERT model download
```
cd modules
wget https://github.com/twigfarm/letr-sol-Ko-UniVL/releases/download/v0/koBERT.zip
unzip koBERT.zip
cd ..
```
#### Tokenizer download 
```
git init
git config core.sparseCheckout true
git remote add -f origin https://github.com/monologg/KoBERT-Transformers.git
echo "kobert_transformers/*" >> .git/info/sparse-checkout
git pull origin master
```


## Original(Eng) Datasets
- 한국어 번역되지 않은 UniVL 학습에 사용된 데이터들은 모두 [UniVL](https://github.com/microsoft/UniVL)에 release 되어있습니다.
  - HowTo100M pretrained binary file
  - YouCook2
  - MSR-VTT
- 원본 HowTo100M Dataset에 대한 feature file은 여기에서 다운로드할 수 있습니다. ([HowTo100M](https://www.di.ens.fr/willow/research/howto100m/))


## Pretraining data
`한국어 HowTo100M 데이터는 사전 학습 파일만 제공합니다`
#### Korean HowTo100M Dataset
- 약 1억 3천만 개의 유튜브 동영상 데이터와 그에 대한 캡션으로 구성된 오픈 도메인 데이터셋
- 유튜브 자막과 나레이션을 캡션으로 사용
- 1억 3천만 개 영상 중 약 1만여개의 동영상 데이터 사용
  - Data Preprocessing 과정은 [UniVL dataloaders](https://github.com/microsoft/UniVL/tree/main/dataloaders), [VideoFeatureExtractor](https://github.com/ArrowLuo/VideoFeatureExtractor)를 참고
- Google translation API와 LETR API를 사용해 약 1만여개의 영상에 대한 약 80만 개의 캡션을 한국어로 번역
- 데이터 구조
```
'O9QQzbxiIVs': {
                'start': array([3.62, 6.27, 7.56, ..., 46.77, 49.37, 51.44], dtype=object),
                'end': array([7.56, 10.2, 12.36, ..., 51.44, 53.21, 56.31], dtype=object),
                'text': array(['실제 스레드를 그렸습니다', '차이 플러스', '행크 애벌레 노래', ..., '품질 인쇄 어', '진짜 스레드 물', '베이스'], dtype=object)
                }
```
- 다운로드
```
mkdir -p ./weight
wget -P ./weight https://github.com/twigfarm/letr-sol-Ko-UniVL/releases/download/v0/pytorch_model.bin.pretrain.49
```



## Finetuning data
#### Korean YouCook2 Dataset
- 1702개의 유튜브 동영상 데이터와 그에 대한 캡션으로 구성된 요리 도메인 데이터셋
- LETR API를 사용해 약 1만 3천여개의 캡션을 한국어로 번역
- Original UniVL의 youcook_no_transcription 파일 사용
- 데이터 구조
```
'nuwCjQVlBrg': {
                'start': array([ 73,  87,  90, 131, 156, 243, 315, 380]),
                'end': array([ 80,  89, 124, 148, 167, 275, 373, 400]),
                'text': array(['붉은 양파를 잘게 썰어 그릇에 담다', '딜을 잘게 썰어 그릇에 담다', '볼에 설탕 소금 식초 오일 스톡과 베이컨 지방을 넣는다.', '감자 껍질을 벗기다', '감자를 잘게 썰어 그릇에 담다', '고기를 밀가루 계란과 빵가루에 묻히다.', '돼지기름을 프라이팬에 녹이다', '고기를 기름에 튀기다'], dtype=object),
                'transcript': array(['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'], dtype=object)
                }
```
- 다운로드
```
mkdir -p data
cd data
wget https://github.com/twigfarm/letr-sol-Ko-UniVL/releases/download/v0/youcookii.zip
unzip youcookii.zip
cd ..
```

#### Korean MSR-VTT Dataset
- 20만 개의 유튜브 동영상 데이터와 그에 대한 캡션으로 구성된 오픈 도메인 데이터셋
- LETR API를 사용해 약 20만  캡션을 한국어로 번역
- 다운로드
```
mkdir -p data
cd data
wget https://github.com/twigfarm/letr-sol-Ko-UniVL/releases/download/v0/msrvtt.zip
unzip msrvtt.zip
cd ..
```


## Download pretrained weight from UniVL
1억 3천만 개의 영상의 HowTo100M 영어 데이터셋으로 Original UniVL 을 사전학습 한 모델 파일입니다.
해당 Weight file을 사용할 경우 가장 좋은 성능의 한국어 캡션을 생성할 수 있습니다.

```
mkdir -p ./weight
wget -P ./weight https://github.com/microsoft/UniVL/releases/download/v0/univl.pretrained.bin
```



## Pretraining Process for Korean HowTo100M dataset
한국어 HowTo100M 데이터셋 사전 학습 과정(제공된 weight 미사용 버전)
`직접 데이터 전처리 및 사전 학습 진행 시 진행 과정입니다.`

#### Requirements
- `HowTo100m.csv` : video id와 feature file을 column으로 갖는 csv file
- `features` : Input 동영상들의 S3D feature files들이 저장된 폴더 (feature 추출 방법 : [VideoFeatureExtractor](https://github.com/ArrowLuo/VideoFeatureExtractor))
- `caption.pickle` : Input 동영상들에 대한 caption들과 그 외 정보들이 dictionary 형태로 저장된 pickle file

#### Format of csv
```
video_id,feature_file
Z8xhli297v8,Z8xhli297v8.npy
...
```

#### Stage1
`Batch size는 본인의 학습 환경에 맞게 조절해주세요`
```
ROOT_PATH=.
DATA_PATH=${ROOT_PATH}/data
SAVE_PATH=${ROOT_PATH}/models
MODEL_PATH=${ROOT_PATH}/UniVL
python -m torch.distributed.launch --nproc_per_node=1 \
${MODEL_PATH}/main_pretrain.py \
 --do_pretrain --num_thread_reader=0 --epochs=50 \
--batch_size=480 --n_pair=3 --n_display=100 \
--bert_model koBERT --do_lower_case --lr 1e-4 \
--max_words 48 --max_frames 64 --batch_size_val 86 \
--output_dir ${SAVE_PATH}/pre_trained/L48_V6_D3_Phase1 \
--features_path ${DATA_PATH}/features \
--train_csv ${DATA_PATH}/HowTo100M.csv \
--data_path ${DATA_PATH}/caption.pickle \
--visual_num_hidden_layers 6 --gradient_accumulation_steps 16 \
--sampled_use_mil --load_checkpoint
```

#### Stage2
`Batch size는 본인의 학습 환경에 맞게 조절해주세요`
- 반드시 Stage1이 선행되어야 합니다
```
ROOT_PATH=.
DATA_PATH=${ROOT_PATH}/data
SAVE_PATH=${ROOT_PATH}/models
MODEL_PATH=${ROOT_PATH}/UniVL
python -m torch.distributed.launch --nproc_per_node=1 \
${MODEL_PATH}/main_pretrain.py \
--do_pretrain --num_thread_reader=0 --epochs=50 \
--batch_size=256 --n_pair=3 --n_display=100 \
--bert_model koBERT --do_lower_case --lr 1e-4 \
--max_words 48 --max_frames 64 --batch_size_val 128 \
--output_dir ${SAVE_PATH}/pre_trained/L48_V6_D3_Phase2 \
--features_path ${DATA_PATH}/features \
--train_csv ${DATA_PATH}/HowTo100M.csv \
--data_path ${DATA_PATH}/caption.pickle \
--visual_num_hidden_layers 6 --decoder_num_hidden_layers 3 \
--gradient_accumulation_steps 60 \
--stage_two --sampled_use_mil \
--pretrain_enhance_vmodal \
--load_checkpoint --init_model ${SAVE_PATH}/pre_trained/L48_V6_D3_Phase1/pytorch_model.bin.pretrain.49
```


## FineTune & Evaluation based on Korean HowTo100M pretrained file
한국어 번역한 HowTo100M 영상 데이터를 베이스로 전이 학습
- *한국어 HowTo100M 사전학습 단계가 선행되거나 사전학습 weight file이 존재해야 합니다*

#### YouCook2
`Batch size는 본인의 학습 환경에 맞게 조절해주세요`
```
TRAIN_CSV="data/youcookii/youcookii_train.csv"
VAL_CSV="data/youcookii/youcookii_val.csv"
DATA_PATH="data/youcookii/youcookii_data_kor.pickle"
FEATURES_PATH="data/youcookii/youcookii_videos_features.pickle"
INIT_MODEL=<pretrained weight file path>
OUTPUT_ROOT="ckpts"
python -m torch.distributed.launch --nproc_per_node=1 \
main_task_caption.py \
--do_train --num_thread_reader=4 \
--epochs=5 --batch_size=16 \
--n_display=100 \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_youcook_caption --bert_model koBERT \
--do_lower_case --lr 3e-5 --max_words 128 --max_frames 96 \
--batch_size_val 64 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 --stage_two \
--init_model ${INIT_MODEL}
```

> 학습 결과
```
BLEU_1: 0.2670, BLEU_2: 0.1849, BLEU_3: 0.1400, BLEU_4: 0.1078
METEOR: 0.1756, ROUGE_L: 0.3338, CIDEr: 0.8068
```
#### MSR-VTT
`Batch size는 본인의 학습 환경에 맞게 조절해주세요`
```
DATATYPE="msrvtt"
TRAIN_CSV="data/msrvtt/MSRVTT_train.9k.csv"
VAL_CSV="data/msrvtt/MSRVTT_JSFUSION_test.csv"
DATA_PATH="data/msrvtt/MSRVTT_data_kor.json"
FEATURES_PATH="data/msrvtt/msrvtt_videos_features.pickle"
INIT_MODEL=<pretrained weight file path>
OUTPUT_ROOT="ckpts"
python -m torch.distributed.launch --nproc_per_node=1 \
main_task_caption.py \
--do_train --num_thread_reader=4 \
--epochs=5 --batch_size=64 \
--n_display=100 \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_caption --bert_model koBERT \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 16 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 --datatype ${DATATYPE} --stage_two \
--init_model ${INIT_MODEL}
```
> 학습 결과
```
BLEU_1: 0.0238, BLEU_2: 0.0000, BLEU_3: 0.0000, BLEU_4: 0.0000
METEOR: 0.2694, ROUGE_L: 0.0291, CIDEr: 0.0011
```

## FineTune & Evaluation based on Original(English) HowTo100M pretrained file
UniVL이 제공하는 1억 3천만개의 영상에 대한 사전 학습 파일 베이스 전이 학습
- 별도의 사전 학습 과정이 필요하지 않습니다
- 가장 좋은 성능을 보입니다

#### YouCook2
`Batch size는 본인의 학습 환경에 맞게 조절해주세요`
```
TRAIN_CSV="data/youcookii/youcookii_train.csv"
VAL_CSV="data/youcookii/youcookii_val.csv"
DATA_PATH="data/youcookii/youcookii_data_kor.pickle"
FEATURES_PATH="data/youcookii/youcookii_videos_features.pickle"
INIT_MODEL="weight/univl.pretrained.bin"
OUTPUT_ROOT="ckpts"
python -m torch.distributed.launch --nproc_per_node=1 \
main_task_caption.py \
--do_train --num_thread_reader=4 \
--epochs=5 --batch_size=16 \
--n_display=100 \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_youcook_caption --bert_model koBERT \
--do_lower_case --lr 3e-5 --max_words 128 --max_frames 96 \
--batch_size_val 64 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 --stage_two \
--init_model ${INIT_MODEL}
```
> 학습 결과
```
BLEU_1: 0.2979, BLEU_2: 0.2349, BLEU_3: 0.1874, BLEU_4: 0.1508
METEOR: 0.2079, ROUGE_L: 0.3572, CIDEr: 1.2677
```
#### MSR-VTT
`Batch size는 본인의 학습 환경에 맞게 조절해주세요`
```
DATATYPE="msrvtt"
TRAIN_CSV="data/msrvtt/MSRVTT_train.9k.csv"
VAL_CSV="data/msrvtt/MSRVTT_JSFUSION_test.csv"
DATA_PATH="data/msrvtt/MSRVTT_data_kor.json"
FEATURES_PATH="data/msrvtt/msrvtt_videos_features.pickle"
INIT_MODEL="weight/univl.pretrained.bin"
OUTPUT_ROOT="ckpts"
python -m torch.distributed.launch --nproc_per_node=1 \
main_task_caption.py \
--do_train --num_thread_reader=4 \
--epochs=5 --batch_size=64 \
--n_display=100 \
--train_csv ${TRAIN_CSV} \
--val_csv ${VAL_CSV} \
--data_path ${DATA_PATH} \
--features_path ${FEATURES_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt_msrvtt_caption --bert_model koBERT \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 16 --visual_num_hidden_layers 6 \
--decoder_num_hidden_layers 3 --datatype ${DATATYPE} --stage_two \
--init_model ${INIT_MODEL}
```
> 학습 결과
```
BLEU_1: 0.0252, BLEU_2: 0.0000, BLEU_3: 0.0000, BLEU_4: 0.0000
METEOR: 0.2710, ROUGE_L: 0.0300, CIDEr: 0.0013
```
  

## Conclusion
해당 연구에서는 한국어 비디오 캡셔닝 모델 개발을 주제로, 기존 비디오 캡셔닝 SOTA 모델인 UniVL이 적절한 한국어 캡션을 생성해내도록 수정하는 과정을 거쳤습니다.  
그 결과 현재 한국어 토크나이징이 잘 이루어지고 적절한 한국어 캡션이 생성되는 상태이지만,  
한국어 HowTo100M 데이터를 사전 학습에 사용한 경우 데이터의 수가 매우 적어 캡션 생성 결과가 좋지 않은 편입니다.  
반면에 Original HowTo100M weight 파일을 적용해 학습을 진행한 경우 현재로서는 가장 좋은 성능을 보이고 있으나, 기존 UniVL repo에서 제공된 weight 파일이 영어를 기준으로 사전학습된 파일이라는 점에서 최고의 성능을 내는 상태는 아니라고 볼 수 있습니다.  
동일한 크기의 데이터 셋을 한국어 번역해 사전학습한다면 더욱 훌륭한 성능의 한국어 비디어 캡셔닝 모델로서 활용될 수 있을 것입니다.



## Reference
- [UniVL](https://github.com/microsoft/UniVL)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)
- [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/)
- [VideoFeatureExtractor](https://github.com/ArrowLuo/VideoFeatureExtractor)
- [Pytube](https://pytube.io/en/latest/)