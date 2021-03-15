import torch

cuda_available = torch.cuda.is_available()
print('CUDA is available: ' + str(cuda_available))
print('PyTorch version: ' + str(torch.__version__))
if cuda_available:
  torch.device('cuda')

import os
import time
import sys
import json
import numpy as np
import pickle
import shutil

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import classification_report
import simpletransformers
import logging
import pandas as pd

# Set data name and path
data_name = 'gab_no_augment'
data_path = './Augment4Gains/data/gab'

# Expectation:
# data_path directory should contain train, val, test jsons
# data-points should be present as a list of dicts
# with each dict having a 'source', and a 'target'

with open(data_path + '/' + 'train.json', 'r+') as f:
  raw_train = json.load(f)

with open(data_path + '/' + 'val.json', 'r+') as f:
  raw_val = json.load(f)

with open(data_path + '/' + 'test.json', 'r+') as f:
  raw_test = json.load(f)

# Verifying loaded data
assert type(raw_train) == type(raw_val)
assert type(raw_train) == type(raw_test)
print('Raw data object type: ' + str(type(raw_train)))
print()

print('Fields in the raw data: ')
unit = raw_train[0]

for key in unit:
  print('• ' + str(key))

# To test out the procedure with small amounts of data
global_testing_mode = 0
global_testing_unit_count = 512

print('Number of Samples in: ')
print('• train: ' + str(len(raw_train)))
print('• val: ' + str(len(raw_val)))
print('• test: ' + str(len(raw_test)))

# Defining mappins for training
def create_set(set_name = 'train'):
  global raw_train, raw_val, raw_test
  global global_testing_mode, global_testing_unit_count
  work_on = None

  if set_name == 'train':
    work_on = raw_train
  elif set_name == 'val':
    work_on = raw_val
  elif set_name == 'test':
    work_on = raw_test
  else:
    print('Invalid Data Split.')
    return -1
  
  data_size = len(work_on)
  if global_testing_mode:
    data_size = global_testing_unit_count

  data = []
  for index in range(data_size):
    unit = [work_on[index]['source'], work_on[index]['target']]
    data.append(unit)

  return data

train = create_set('train')
val = create_set('val')
test = create_set('test')

# Getting number of positive and negative samples in train split
total_in_train = len(train)
positive_in_train = 0

for unit in train:
  positive_in_train += unit[1]

print('Number of positive samples: ' + str(positive_in_train))
print('Number of negative samples: ' + str(total_in_train - positive_in_train))

# Weights to correct the class imbalance
greater_class_count = max((total_in_train - positive_in_train), positive_in_train)
class_weights = [greater_class_count / (total_in_train - positive_in_train),
                 greater_class_count / positive_in_train]

# Defining dataframes
train_df = pd.DataFrame(train)
train_df.columns = ['source', 'label']

val_df = pd.DataFrame(val)
val_df.columns = ['source', 'label']

# Leveraging a pre-trained Transformer Model

model_index = 0
# Set 0 for bert-base, 1 for roberta-base

model_loc = ['bert-base-uncased', 'roberta-base'][model_index]
model_type = ['bert', 'roberta'][model_index]

is_lower = False
if model_index == 0:
  is_lower = True

length_setting = 256
model_name = model_loc + '_' + data_name + '_' + str(length_setting)
cache_name = model_name + '_cache_dir'

batch_size = 32
num_epochs = 8
num_gpus = 1

if global_testing_mode == 1:
  model_name += '_testing'
  num_epochs = 2
  length_setting = 64

model_args = ClassificationArgs(train_batch_size = batch_size,
                                max_seq_length = length_setting,
                                save_steps = -1,
                                n_gpu = num_gpus,
                                num_train_epochs = num_epochs,
                                evaluate_during_training = True,
                                overwrite_output_dir = True,
                                save_eval_checkpoints = False,
                                save_model_every_epoch = False,
                                cache_dir = cache_name,
                                fp16 = True,
                                manual_seed = 42,
                                do_lower_case = is_lower,
                                best_model_dir = model_name)

model = ClassificationModel(model_type,
                            model_loc,
                            use_cuda = cuda_available,
                            args = model_args,
                            num_labels = 2,
                            weight = class_weights)

# Training
start = time.time()
model.train_model(train_df, eval_df = val_df)
end = time.time()
time_to_train = int(round(end - start))

hours = int(time_to_train / 3600)
minutes = int(int(time_to_train % 3600) / 60)
seconds = int(time_to_train % 60)
print()
print('Number of Epochs: ' + str(num_epochs))
print('Maximum Sequence Length: ' + str(length_setting))
print('Batch size: ' + str(batch_size))
print('Time taken for training: ' + str(hours).zfill(2) + ':' + str(minutes).zfill(2) + ':' + str(seconds).zfill(2))

# Inference
infer_now = True

if infer_now == True:
  model = ClassificationModel(model_type, model_name)
  print('Using Model: ' + str(model_name))
  print()
  
  val_sources = [unit[0] for unit in val]
  test_sources = [unit[0] for unit in test]

  val_targets = [unit[1] for unit in val]
  test_targets = [unit[1] for unit in test]

  # Evaluation on val data
  print('Results on the validation split: ')
  val_predictions, val_outputs = model.predict(val_sources)
  print(classification_report(val_targets, val_predictions))
  print()

  # Evaluation on test data
  print('Results on the test split: ')
  test_predictions, test_outputs = model.predict(test_sources)
  print(classification_report(test_targets, test_predictions))

compress_model = True
if compress_model == True:
  shutil.make_archive(model_name, 'zip', model_name)
  shutil.make_archive(cache_name, 'zip', cache_name)

# ^_^ Thank You