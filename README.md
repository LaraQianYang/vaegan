# vaegan

## Model:
This is the implementation of [An End-to-End Generative Architecture for Paraphrase Generation](https://www.aclweb.org/anthology/D19-1309.pdf) (EMNLP 2019).


## Usage
### Before model training, it is necessary to download the datasets. 
```
  --train_source_path=quora/100k/train_source.txt \
  --train_target_path=quora/100k/train_target.txt \
  --valid_source_path=quora/100k/test_source.txt \
  --valid_target_path=quora/100k/test_target.txt \
  --info_dir=quora/100k/info/ 
```

### Install pycocoevalcap package. 

### To train the model, use:
```
$ sh demo_run.sh
```

### To achieve better performance, please download the glove embeddings.

