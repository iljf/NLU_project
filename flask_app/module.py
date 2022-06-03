from transformers import AutoTokenizer
from transformers import RobertaForSequenceClassification
import torch
import re

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
class MODEL():
    def __init__(self, model_path, model_name):
        # load checkpoint file
        # Load trained model
        self.model_path = model_path
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.model.load_state_dict(torch.load("C:/Users/KDB/Desktop/CP2/model_v4.ckpt.3",  map_location=device)["model_state_dict"], strict=False)
        #load tokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    def forward(self, sentence1, sentence2):
        tokens = self.tokenizer(sentence1, sentence2,padding=True, truncation=True, return_tensors="pt")
        output = self.model(**tokens)
        return output

def pre_treat(sent):
    sentence = sent.strip()
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    sentence = hangul.sub('', sentence)
    return sentence