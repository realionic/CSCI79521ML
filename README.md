# CSCI79521ML

## Model
BertForSequenceClassification. \
This is a transformer model with a sequence classification/regression head on top.

### Hyperparameters
hyperparameters for all models are set to be the same. \
learning rate: 1e-5 \
batch size: 32 \
All models are trained on a single v100 GPU

## Data
- goEmotions
  - This dataset contains 27 labels - amusement, etc.
  - Dataset size: train(), validation(), test() 
- Toxicity
  - Originally the dataset is multi-labelled with six labels (toxic, severe_toxic, obscene, neutral, etc.). However, to make the task simpler, the task is conducted as a uni-labelled, binary classification with only the labels that contain "toxic" are used to represent the toxicity.
  - This dataset didn't come with validation, so 20% of train set was used as a validation set.
  - Dataset size: train(), validation(), test() 

## Result
| Model | pre-training data | fine-tuninig data | F1 Score | 
| - | - | - | -|
| bert-base-uncased | - | goEmotions | 0.613 |
| bert-base-cased | - | goEmotions | 0.608 |
| bert-base-cased | - | toxicity | 0.934 |
| bert-base-cased | goEmotions | toxicity | - |
| bert-base-cased | toxicity | goEmotions | - |
