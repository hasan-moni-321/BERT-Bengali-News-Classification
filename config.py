import pandas as pd 
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch 


def dataset():
    dataset = pd.read_csv("/home/hasan/Desktop/Code to keep on Github/BERT Bengali News Classification/clean_news_classification.csv", usecols=["category", "content"])
    for cat in ['education', 'durporobash', 'northamerica', 'pachmisheli', 'weare', 'onnoalo', 'roshalo', 'bondhushava', 'specialsupplement', 'kishoralo', 'trust', 'protichinta', '1', 'nagorikkantho', 'chakribakri', 'tarunno', 'mpaward1', '22221', 'facebook', 'events', 'diverse', 'democontent', 'democontent', 'AskEditor', 'bsevents']:
        dataset = dataset.drop(dataset[dataset['category']==cat].index)
    return dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 5
max_len = 512
batch_size = 4

model_path = "/home/hasan/Desktop/Code to keep on Github/BERT Bengali News Classification/Bengali_bert.bin"

bert_model = "sagorsarker/bangla-bert-base"
tokenizer = AutoTokenizer.from_pretrained(bert_model)
#model = AutoModelForMaskedLM.from_pretrained(bert_model) 

