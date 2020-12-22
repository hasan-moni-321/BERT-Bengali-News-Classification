import numpy as np
import pandas as pd 
from collections import defaultdict

from sklearn.utils import shuffle

import torch 
from torch.utils.data import DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup


import config 
import dataset 
import model
import engine 


# Reading dataset
df = config.dataset()


# Shuffle dataset
df = shuffle(df)
print(df.head()) 

# Categorical to Numerical
print(df['category'].value_counts())


d = {"bangladesh": 1, "sports": 2, "international": 3, "entertainment": 4, "economy": 5, "opinion": 6, "technology": 7, "lifestyle": 8}
df['sentiment'] = df['sentiment'].map(d)
#df_class2 = df['sentiment'].value_counts()


# Dividing dataset
df_train, df_test = train_test_split(df, test_size=0.3, random_state=101)
df_val, df_test = train_test_split(df_test, test_size=0.2, random_state=101)
#print(f"Shape of df_train is {df_train.shape}, Shape of df_val is {df_val.shape}, Shape of df_test is {df_test.shape}")


# Loading Datasets using custom loader
def create_data_loader(df, tokenizer, max_len, batch_size):

    ds = dataset.BERTDataset(
                            reviews=df.review.to_numpy(),
                            targets=df.sentiment.to_numpy(),
                            tokenizer=tokenizer,
                            max_len=max_len
                            )

    return DataLoader(
                      ds,
                      batch_size=batch_size,
                      num_workers=4
                      )


train_data_loader = create_data_loader(df_train, config.tokenizer, config.max_len, config.batch_size)
val_data_loader = create_data_loader(df_val, config.tokenizer, config.max_len, config.batch_size)
test_data_loader = create_data_loader(df_test, config.tokenizer, config.max_len, config.batch_size)


# Model
device = config.device
EPOCHS = config.epoch
model = model.BERTBaseUncased()
model.to(device) 


# Hyperparameter
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
                                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
                               ] 


optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )


best_accuracy = 0
for epoch in range(config.EPOCHS):
    engine.train_fn(train_data_loader, model, optimizer, device, scheduler, epoch)
    outputs, targets = engine.eval_fn(valid_data_loader, model, device)

    outputs = np.array(targets) >= 0.5 

    accuracy = metrics.roc_auc_score(targets, outputs)
    accuracy2 = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')

    print(f"Epoch = {epoch}, roc_auc Score = {accuracy}")
    print(f"Epoch = {epoch}, Accuracy Score = {accuracy2}")
    print(f"Epoch = {epoch}, f1_micro Score = {f1_score_micro}")
    print(f"Epoch = {epoch}, f1_macro Score = {f1_score_macro}")

    if accuracy > best_accuracy:
        torch.save(model.state_dict(), config.model_path) 
        best_accuracy = accuracy



######################################################
# Model Evaluation
######################################################
model = model.load_state_dict(torch.load(config.model_path))

# Predicting with test data
epoch = 1
for epo in range(epoch):
    engine.test_fn(test_data_loader, model, device)



#######################################################
# Custom Prediction
#######################################################
text = "Bangladesh forms the larger and eastern part of the Bengal region.[15] According to the ancient Indian texts, Rāmāyana and Mahābhārata, the Vanga Kingdom, one of the namesakes of the Bengal region, was a strong naval ally of the legendary Ayodhya. In the ancient and classical periods of the Indian subcontinent, the territory was home to many principalities, including the Pundra, Gangaridai, Gauda, Samatata, and Harikela. It was also a Mauryan province under the reign of Ashoka. The principalities were notable for their overseas trade, contacts with the Roman world, the export of fine muslin and silk to the Middle East, and spreading of philosophy and art to Southeast Asia. The Pala Empire, the Chandra dynasty, and the Sena dynasty were the last pre-Islamic Bengali middle kingdoms. Islam was introduced during the Pala Empire, through trade with the Abbāsid Caliphate,[16] but following the early conquest of Bakhtiyar Khalji and the subsequent establishment of the Delhi Sultanate and preaching of Shah Jalāl in East Bengal, the faithfully spread across the region. In 1576, the wealthy Bengal Sultanate was absorbed into the Mughal Empire, but its rule was briefly interrupted by the Suri Empire. Following the death of Emperor Aurangzeb in the early 1700s, proto-industrialized Mughal Bengal became a semi-independent state under the Nawabs of Bengal. The region was later conquered by the British East India Company at the Battle of Plassey in 1757"

engine.predict_custom(text, model):

