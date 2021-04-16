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
import random

from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import classification_report
import simpletransformers
import logging
import pandas as pd

augment_to = 15000

with open('./config.json', 'r+') as f:
  mappings = json.load(f)

# Assignments
data_name = mappings['name']
data_path = mappings['path']

# Name of the experiment
global_testing_mode = mappings['test_mode']
# Flag for testing out an implementation

is_augment = mappings['use_aug']
augment_source = mappings['aug_src']

model_index = mappings['model']
# Set 0 for bert-base-uncased, 1 for roberta-base

correct_imbalance = mappings['bal']
# To explicitly use weights to correct class imbalance

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

if is_augment == 1:
  with open(data_path + '/' + augment_source, 'r+') as f:
    raw_aug = json.load(f)

# Verifying loaded data
assert type(raw_train) == type(raw_val)
assert type(raw_train) == type(raw_test)

global_testing_unit_count = 512

print('Number of Samples in: ')
print('• train: ' + str(len(raw_train)))
print('• val: ' + str(len(raw_val)))
print('• test: ' + str(len(raw_test)))
if is_augment == 1:
  print('• augment: ' + str(len(raw_aug)))

# Defining mappings for training
def create_set(set_name = 'train'):
  global raw_train, raw_val, raw_test, raw_aug
  global global_testing_mode, global_testing_unit_count
  global is_augment, augment_to
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

  if set_name == 'train' and is_augment == 1:
    positives = 0
    negatives = 0

    for unit in data:
      if unit[1] == 1:
        positives += 1
      else:
        negatives += 1

    need_positives = max(augment_to - positives, 0)
    need_negatives = max(augment_to - negatives, 0)

    work_on = raw_aug
    data_size = len(work_on)
    
    zeros = []
    ones = []
    
    for index in range(data_size):
      unit = [work_on[index]['source'], work_on[index]['target']]
      if unit[1] == 1:
        ones.append(unit)
      else:
        zeros.append(unit)

    use_positives = min(len(ones), need_positives)
    use_negatives = min(len(zeros), need_negatives)

    random.seed(42)
    random.shuffle(ones)
    random.shuffle(zeros)

    augment_collection = ones[0: use_positives] + zeros[0: use_negatives]
    random.shuffle(augment_collection)

    for unit in augment_collection:
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

if correct_imbalance == 0:
  # Disabling weighing of classes
  class_weights = [1, 1]

elif correct_imbalance == 1:
  # Weights to correct the class imbalance
  # First Method
  greater_class_count = max((total_in_train - positive_in_train), positive_in_train)
  class_weights = [greater_class_count / (total_in_train - positive_in_train),
                   greater_class_count / positive_in_train]
else:
  # Weights to correct the class imbalance
  # Second Method
  class_weights = [total_in_train / (2 * (total_in_train - positive_in_train)),
                   total_in_train / (2 * positive_in_train)]

# Defining dataframes
train_df = pd.DataFrame(train)
train_df.columns = ['source', 'label']

val_df = pd.DataFrame(val)
val_df.columns = ['source', 'label']

# Leveraging a pre-trained Transformer Model

model_loc = ['bert-base-uncased', 'roberta-base'][model_index]
model_type = ['bert', 'roberta'][model_index]

is_lower = False
if model_index == 0:
  is_lower = True

length_setting = 256
model_name = model_loc + '_' + data_name + '_' + str(length_setting)
cache_name = model_name + '_cache_dir'

batch_size = 128
num_epochs = 4
num_gpus = 4

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
  print(classification_report(val_targets, val_predictions, digits = 6))
  print()

  # Evaluation on test data
  print('Results on the test split: ')
  test_predictions, test_outputs = model.predict(test_sources)
  print(classification_report(test_targets, test_predictions, digits = 6))

remove_backup = True
if remove_backup == True:
  os.system('rm -rf ' + cache_name)
  os.system('rm -rf ' + model_name)
  os.system('rm -rf outputs')

compress_model = False
if compress_model == True:
  shutil.make_archive(model_name, 'zip', model_name)
  shutil.make_archive(cache_name, 'zip', cache_name)
