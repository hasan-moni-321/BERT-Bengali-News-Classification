import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM

import config


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(config.bert_model) 
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 8)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(o.2)
        output = self.out(bo)
        return output
