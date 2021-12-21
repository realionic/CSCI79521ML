# -*- coding: utf-8 -*-

# import packages
import torch
import random

import numpy as np
import pandas as pd 

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm



LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'neutral']
RANDOM_SEED = 87
BATCH_SIZE = 32
EPOCHS = 10
num_labels = 2
lr = 1e-5
# seed random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def f1_score_func(preds, labels):

    # Setting up the preds to axis=1
    # Flatting it to a single iterable list of array
    preds_flat = np.argmax(preds, axis=1).flatten()

    # Flattening the labels
    labels_flat = labels.flatten()

    # Returning the f1_score as define by sklearn
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels, LABELS): 
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    # Iterating over all the unique labels
    # label_flat are the --> True labels
    for label in np.unique(labels_flat):
        # Taking out all the pred_flat where the True alable is the lable we care about.
        # e.g. for the label Happy -- we Takes all Prediction for true happy flag
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {LABELS[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_val):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2]}

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

"""## goEmotions training
### prep data
"""

print("loading the preprocessed data, splitting train data to create validation data (20%)")
print(f'batchsize: {BATCH_SIZE}, num labels: {num_labels}, lr: {lr}')
# load data
data_dir = ""
save_path = ""

df_train = pd.read_csv(data_dir + '/train_pp.csv', header=0)

df_train, df_val = train_test_split(df_train, random_state=RANDOM_SEED, test_size=0.2)
df_test = pd.read_csv(data_dir + '/test_pp.csv', header=0)
df_test_y = pd.read_csv(data_dir + '/test_y_pp.csv', header=0)

# y_dict = {0: 1, 1: 2, 5: 4, 17: 5, 21: 0, 23: 6, 53: 3}
# df_test_y = df_test_y.replace({'sum': y_dict})
# df_val_y = df_val_y.replace({'sum': y_dict})


X_train = df_train['comment_text']
y_train = df_train['toxic'].astype(np.int)
X_val = df_val['comment_text']
y_val = df_val['toxic'].astype(np.int)
X_test = df_test['comment_text']
y_test = df_test_y['toxic'].astype(np.int)


"""### tokenize"""

# encode data
encoded_data_train = tokenizer.batch_encode_plus(
    X_train.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    truncation=True,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    X_val.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    truncation=True,
    return_tensors='pt'
)

# get tensor dataset
print('Get the dataset....')
dataset_train = TensorDataset(encoded_data_train['input_ids'], 
                                     encoded_data_train['attention_mask'], 
                                     torch.tensor(y_train.values))

dataset_validation = TensorDataset(encoded_data_val['input_ids'], 
                                     encoded_data_val['attention_mask'],
                                     torch.tensor(y_val.values))

# get dataloder
dataloader_train = DataLoader(dataset_train, 
                                   sampler=RandomSampler(dataset_train),
                                   batch_size=BATCH_SIZE)

dataloader_validation = DataLoader(dataset_validation, 
                                   sampler=RandomSampler(dataset_validation),
                                   batch_size=BATCH_SIZE)

"""### prep train"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print('Initiating the model...')
model = BertForSequenceClassification.from_pretrained("bert-base-cased",
                                                    num_labels=num_labels,
                                                    output_attentions = False,
                                                    output_hidden_states = False)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train) * EPOCHS)

log = [['epoch', 'loss_train_avg', 'val_loss', 'val_f1']]
best_val_loss = np.inf
early_stop_threshold = 4
early_stop_count = 0

for epoch in tqdm(range(1, EPOCHS+1)):
    model.train()          # Sending our model in Training mode
    loss_train_total = 0   # Setting the training loss to zero initially

    # Setting up the Progress bar to Moniter the progress of training
    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad() # As we not working with thew RNN's
        
        # As our dataloader has '3' iteams so batches will be the Tuple of '3'
        batch = tuple(b.to(device) for b in batch)
        
        # INPUTS
        # Pulling out the inputs in the form of dictionary
        inputs = {
            'input_ids':      batch[0],
            'attention_mask': batch[1],
            'labels':         batch[2],
        }

        outputs = model(inputs['input_ids'],
                        token_type_ids=None,
                        attention_mask=inputs['attention_mask'],
                        labels=inputs["labels"])


        # OUTPUTS
        outputs = model(**inputs) # '**' Unpacking the dictionary stright into the input
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()           # backpropagation

        # Gradient Clipping -- Taking the Grad. & gives it a NORM value ~ 1 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    loss_train_avg = loss_train_total/len(dataloader_train)            
    val_loss, predictions, true_vals = evaluate(dataloader_validation)

    val_f1 = f1_score_func(predictions, true_vals)

    tqdm.write(f'\nEpoch {epoch}')
    tqdm.write(f'Training loss: {loss_train_avg}')
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')

    log.append([str(epoch), str(loss_train_avg), str(val_loss), str(val_f1)])

    print(f"validation loss: {val_loss}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"new validation loss is found: {best_val_loss}")
        torch.save(model.state_dict(), f'{save_path}/toxicity_preprocessed_bestmodel.pt')
        # reset the early stopping count
        early_stop_count = 0
    
    # employ early stopping: if the val loss doesn't improve over 4 consecutive epochs, stop training
    else:
        early_stop_count += 1
        if early_stop_count > early_stop_threshold:
            break

file = open(f'{save_path}/toxicity_preprocessed_log.csv', 'w')

for line in log:
    file.write(','.join(line) + '\n')

file.close()
