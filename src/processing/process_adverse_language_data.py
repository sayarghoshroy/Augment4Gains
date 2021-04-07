# Processing dataset from 'Detecting Online Hate Speech Using Context Aware Models'
# Raw data can be found at:
# github.com/t-davidson/hate-speech-and-offensive-language

import preprocessor as tweet_proc

import csv
import json
import random

file_name = 'labeled_data.csv'
see_index = True

data = []

file = open(file_name, 'r')

file_reader = csv.reader(file, delimiter = ",")
for line in file_reader:
  # line[6] contains the Tweet text
  # line[5] contains the Tweet label
  if see_index == True:
    see_index = False
    continue
  unit = {}
  base = str(line[6])
  cleaned_source = tweet_proc.clean(base.replace('\n', ' '))
  unit['source'] = cleaned_source
  
  if str(line[5]).strip() == '2':
    unit['target'] = 0
  else:
    unit['target'] = 1
  data.append(unit)

indices = [id for id in range(len(data))]
random.seed(42)
random.shuffle(indices)

train_size = int((7 / 10) * len(data))
val_size = int((1 / 10) * len(data))
test_size = len(data) - (train_size + val_size)

print('Train Set Size: ' + str(train_size))
print('Validation Set Size: ' + str(val_size))
print('Test Set Size: ' + str(test_size))

train_indices = indices[0: train_size]
val_indices = indices[train_size: train_size + val_size]
test_indices = indices[train_size + val_size: ]

# Defining the splits

train, val, test = [], [], []

for index in indices:
  if index in train_indices:
    train.append(data[index])
  elif index in val_indices:
    val.append(data[index])
  else:
    test.append(data[index])

# Splits created

# Saving the splits
with open('train.json', 'w+') as f:
  json.dump(train, f)

with open('val.json', 'w+') as f:
  json.dump(val, f)

with open('test.json', 'w+') as f:
  json.dump(test, f)

# ^_^ Thank You