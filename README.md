# KiL
The source code of "Improving Text Classification with Knowledge in Labels" is provided here.

**PS: The model is trained on a Tesla V100 32GB gpu. For the reproducibility, please use the same device.**

# Usage

## 1. Environment settings
### 1. First install libraries 
```
$ pip install -r requirements.txt
```

### 2. download glove embeddings
please download pretrained embeddings from [here](https://nlp.stanford.edu/projects/glove/). Select `glove.6B.zip` and extract `glove.6B.300d.txt` here.


## 2. Training LSTM model

You can choose dataset from {agnews, imdb, newsgroup} and select your gpu id.
```
$ python kil_lstm.py -d newsgroup -g 0
```

## 3. Training BERT model
You can choose dataset from {agnews, imdb, newsgroup} and select your gpu id.
```
$ CUDA_VISIBLE_DEVICES=0 python kil_bert.py -d newsgroup
```
