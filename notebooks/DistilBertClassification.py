import torch
from logging import warning
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


import numpy as np
import pandas as pd
import os
import shutil
import time
import datetime

import warnings
warnings.filterwarnings('ignore')

if torch.cuda.is_available():     
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Initiate loss to Binary Cross Entropy
criterion = nn.BCELoss()

# DistilBERT Model 1 hidden layer
class BertClassification(nn.Module):
    def __init__(self, output_dim=1, hidden_size=384, dropout=0.2):
        super(BertClassification, self).__init__()
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.bert_hidden_size = self.bert_model.config.hidden_size
        self.hidden_size = hidden_size
        self.hidden_layer = nn.Linear(self.bert_hidden_size, self.hidden_size)

        self.drop_out = nn.Dropout(dropout)

        self.classification = nn.Linear(self.hidden_size, output_dim)

        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_model(input_ids=input_ids,
                                      attention_mask=attention_mask)

        cls_token = bert_output[0][:, 0]

        hidden_ouput = self.ReLU(self.hidden_layer(cls_token))

        hidden_ouput = self.drop_out(hidden_ouput)
        output = self.Sigmoid(self.classification(hidden_ouput))
        output = output.flatten()

        return output
        
        

# DistilBERT Model 2 hidden layers        
class BertClassificationML(nn.Module):
    """"DistilBERT with 2 hidden layers and option to unfreeze the last transformer layer"""
    def __init__(self,output_dim=1, hidden_size=256, hidden_size2=32, dropout=0.1, unfreeze=False):
        super(BertClassificationML, self).__init__()
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        # Unfreeze the last DistilBERT transformer layer
        if unfreeze:
            for name, param in self.bert_model.named_parameters():
                if 'transformer.layer.5.' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.bert_hidden_size = self.bert_model.config.hidden_size
        self.hidden_size = hidden_size
        self.hidden_size2 = hidden_size2
        self.hidden_layer = nn.Linear(self.bert_hidden_size, self.hidden_size)
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size2)

        self.drop_out = nn.Dropout(dropout)

        self.classification = nn.Linear(self.hidden_size2, output_dim)

        self.GeLU = nn.GELU()
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_model(input_ids=input_ids, 
                                      attention_mask=attention_mask)
        
        # hidden_state = bert_output[0]
        cls_token = bert_output[0][:,0]

        hidden_ouput = self.GeLU(self.hidden_layer(cls_token))
        hidden_ouput = self.drop_out(hidden_ouput)
        hidden_ouput = self.GeLU(self.hidden_layer2(hidden_ouput))
        hidden_ouput = self.drop_out(hidden_ouput)

        output = self.Sigmoid(self.classification(hidden_ouput))
        output = output.flatten()

        return output

		
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(round(seconds))))


## Create data loader for inference
def NonToxicScoreDataLoader(output_file, output_col):
    output_df = pd.read_csv(output_file, sep="\t")

    # Load DistilBERT tokenizer
    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Max_length used from pretrained model
    max_length = 64

    output_encodings = bert_tokenizer(
        list(output_df[output_col].values), 
        max_length=max_length,
        truncation=True,
        pad_to_max_length=True, 
        return_tensors='pt'
    )

    input_ids = output_encodings.input_ids.to(device, dtype = torch.long)
    attention_mask = output_encodings.attention_mask.to(device, dtype = torch.long)
    # labels = torch.zeros(output_df[output_col].shape[0])

    output_dataset = TensorDataset(input_ids, attention_mask)
    output_loader = DataLoader(output_dataset, batch_size=16)


    return output_loader


## Calculate NonToxicScore function
def NonToxicScore(output_loader, model, verbose=True):
    # Initialize
    outputs=[]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for step, data in enumerate(output_loader):
            
            # send the data to cuda device
            ids = data[0].to(device, dtype = torch.long)
            mask = data[1].to(device, dtype = torch.long)

            # compute output / NonToxicScore
            output = model(input_ids=ids,
                           attention_mask=mask)

            outputs.extend(output.cpu().detach().numpy().tolist())
      

    avg_NonToxicScore = np.mean(outputs)
    metrics = {"NonToxicScore": avg_NonToxicScore}
    # elapsed_time = time.time() - t0
    if verbose:
      print(metrics)
    
    return outputs, metrics
  
	
## Define validate function
def validate(val_loader, model, criterion=criterion, verbose=False):
    # Initialize
    targets=[]
    outputs=[]
    t0 = time.time()
    total_val_loss = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for step, data in enumerate(val_loader):
            
            # send the data to cuda device
            ids = data[0].to(device, dtype = torch.long)
            mask = data[1].to(device, dtype = torch.long)
            target = data[2].float().to(device)

            # compute output
            output = model(input_ids=ids,
                           attention_mask=mask)

            # compute loss
            loss = criterion(output, target)

            # Accumulate the validation loss over all of the batches to calculate average loss
            total_val_loss += loss.item()

            targets.extend(target.cpu().detach().numpy().tolist())
            outputs.extend(output.cpu().detach().numpy().tolist())

            if step % 100 == 0 and not step == 0 and verbose==True:
                print('Batch {:>5,}  of  {:>5,}.  Loss {:0.4f}  Elapsed: {:}.'.format(
                    step, 
                    len(val_loader),
                    loss.item(),
                    format_time(time.time() - t0)))        

    avg_val_loss = total_val_loss / len(val_loader)
    elapsed_time = time.time() - t0
    if verbose:
      print ("\nAvg Validation Loss {:0.4f}, Completed in {:}".format(
          avg_val_loss,
          format_time(elapsed_time)
            ))
    
    return outputs, targets, avg_val_loss
	


def compute_metrics(outputs, targets):
    y_pred = (np.array(outputs) >= 0.5).astype(float)
    y_true = np.array(targets)

    # metrics
    f1 = f1_score(y_true= y_true, y_pred=y_pred)
    # roc_auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print ("Accuracy {:0.4f} \n".format(accuracy))
    
    return accuracy
