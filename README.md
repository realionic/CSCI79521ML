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
- goEmotions [paper](https://arxiv.org/abs/2005.00547).
  - This dataset contains 28 labels - amusement, admiration, anger, neutral, etc.
  - Dataset size: train(36,308), validation(4,548), test(4,590) 
- Toxicity [dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
  - Originally the dataset is multi-labelled with six labels (toxic, severe_toxic, obscene, neutral, etc.). However, to make the task simpler, the task is conducted as a uni-labelled, binary classification with only the labels that contain "toxic" are used to represent the toxicity.
  - This dataset didn't come with validation, so 20% of train set was used as a validation set.
  - Dataset size: train(126,913), validation(31,728), test(63,826) 

## Dependencies
- pytorch, numpy, pandas, sklearn, transformers, tqdm, sys

## Run
- To train the model first you have to change data_path (where your data is located) and save_path (where you want your log and model to be saved) within goemotions.py and toxicity.py \
`python goemotions.py` or `python toxicity.py`
- To test the trained model, you need to pass an argument <model_path> where the model directory where your trained model checkpoint is stored. \
`python goemotions_test.py <model_path>` or `python toxicity_test.py <model_path>`

## Result
| Model | pre-training data | fine-tuninig data | F1 Score | 
| - | - | - | -|
| Demszky et al.* | - | goEmotions | 0.46 | 
| bert-base-uncased | - | goEmotions | 0.613 |
| bert-base-cased | - | goEmotions | 0.608 |
| bert-base-cased | toxicity | goEmotions | 0.607 |
| bert-base-cased | - | toxicity | 0.934 |
| bert-base-cased | goEmotions | toxicity | 0.933 |

\* Demszky, Dorottya et al. “GoEmotions: A Dataset of Fine-Grained Emotions.” ArXiv abs/2005.00547 (2020)
