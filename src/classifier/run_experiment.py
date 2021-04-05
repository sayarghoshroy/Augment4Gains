import os
import sys
import json

name = 'anonymous'
path = './data'
test_mode = 0
use_aug = 0
aug_src = 'aug.json'
model = 0
bal = 1

mappings = {
  'name': name,
  'path': path,
  'test_mode': test_mode,
  'use_aug': use_aug,
  'aug_src': aug_src,
  'model': model,
  'bal': bal
}

count = len(sys.argv)

if count % 2 == 2:
  print('Invalid Number of Arguments')
  sys.exit()

arguments = []
for index in range(1, count):
    arguments.append(str(sys.argv[index]))

print('List of arguments: ', flush = True)
print(arguments, flush = True)

try:
  for index in range(len(arguments)):
    if index % 2 == 1:
      continue
    flag = arguments[index]
    value = arguments[index + 1]
    mappings[flag] = value

except:
  print('Invalid Format')
  sys.exit()

try:
  mappings['test_mode'] = int(mappings['test_mode'])
  mappings['use_aug'] = int(mappings['use_aug'])
  mappings['model'] = int(mappings['model'])
  mappings['bal'] = int(mappings['bal'])
except:
  print('Unexpected data type')
  sys.exit()

with open('./config.json', 'w+') as f:
  json.dump(mappings, f)

# Begin experiment
os.system('python classification_test_bench.py')

# Sample Input
# python run_experiment.py name tf_paraphrase path ./data/tf_para test_mode 1 use_aug 1 aug_src aug_tf.json model 1 bal 1
