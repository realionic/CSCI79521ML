import torch
import random

import numpy as np
import pandas as pd 
import sys

from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm


PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
LABELS = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']
RANDOM_SEED = 87
BATCH_SIZE = 32

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

def evaluate(dataloader_val, model, device):

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


# load data
data_path = ""
save_path = ""

df_test = pd.read_csv(data_path + '/test.tsv', sep='\t', names=['text', 'y', 'id'])
df_test = df_test[df_test.apply(lambda x: ',' not in x['y'], axis=1)]

X_test = df_test['text']
y_test = df_test['y'].astype(np.int)

encoded_data_test = tokenizer.batch_encode_plus(
    X_test.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    # max_length=256, 
    return_tensors='pt'
)


dataset_test = TensorDataset(encoded_data_test['input_ids'], 
                                     encoded_data_test['attention_mask'],
                                     torch.tensor(y_test.values))

dataloader_test = DataLoader(dataset_test, 
                                   sampler=RandomSampler(dataset_test),
                                   batch_size=BATCH_SIZE)

device = torch.device('cpu')

model = BertForSequenceClassification.from_pretrained("bert-base-cased",
                                                    num_labels=len(LABELS),
                                                    output_attentions = False,
                                                    output_hidden_states = False)

best_model_path = str(sys.argv[1])
model.load_state_dict(torch.load(best_model_path))
model.to(device)

### test
test_loss, test_preds, test_true_vals = evaluate(dataloader_test, model, device)
test_f1 = f1_score_func(test_preds, test_true_vals)
print(f"test f1 value is {test_f1}")

accuracy_per_class(test_preds, test_true_vals, LABELS)