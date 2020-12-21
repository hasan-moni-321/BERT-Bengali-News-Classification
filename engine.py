import numpy as np 

import torch
import torch.nn as nn
from tqdm import tqdm

from sklearn import metrics 

import config 



def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler, epoch): 
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        if _%5000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets



def test_fn(test_data, model, device):
    outputs, targets = eval_fn(test_data, model, device)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    accuracy_roc_auc = metrics.roc_auc_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')

    print(f"Accuracy Score = {accuracy}")
    print(f"roc_auc Accuracy Score = {accuracy_roc_auc}") 
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")



def predict_custom(text, model, device, max_len):
    review_text = text 
    #max_length = max_len 

    encoded_review = config.tokenizer.encode_plus(
                                    review_text,
                                    None,
                                    add_special_tokens=True,
                                    max_length=max_len,
                                    truncation=True,
                                    pad_to_max_length=True
    )

    ids = encoded_review["input_ids"]
    mask = encoded_review["attention_mask"] 
    token_type_ids = encoded_review["token_type_ids"]

    ids = torch.tensor(ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

    outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
    _, prediction = torch.max(outputs, dim=-1)

    print(f"Sentiment : {}")
