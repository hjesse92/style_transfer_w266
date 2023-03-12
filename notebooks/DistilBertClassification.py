import torch
from logging import warning
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


import numpy as np
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

# DistilBERT Model
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

		
def format_time(seconds):
    return str(datetime.timedelta(seconds=int(round(seconds))))
	
	
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
    # accuracy = np.sum(y_pred == y_true) / len(y_true)

    print ("Accuracy {:0.4f} \n".format(accuracy))
    
    return accuracy
