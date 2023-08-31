## Model Description

- Language model: [XLM-RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html)
- Language: Vietnamese, Englsih
- Dataset (combine English and Vietnamese):
  - [Squad 2.0](https://rajpurkar.github.io/SQuAD-explorer/) 
  - [MultiLingual Question Answering](https://github.com/facebookresearch/MLQA)
  
| Model  | EM | F1 |
| ------------- | ------------- | ------------- |
| [large](https://huggingface.co/xlm-roberta-large)  | 62.5  | 75.97  |


[MRCQA] using [XLM-RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html) as a pre-trained language model. 
## Training model
In data-bin/raw folder already exist some sample data files for the training process. Do following steps:

- Create environment by using file requirements.txt

- Clean data

```shell
python squad_to_mrc.py
python train_valid_split.py
```
- Train model

```shell
python main.py
```

- Test model

```shell
python infer.py
```
