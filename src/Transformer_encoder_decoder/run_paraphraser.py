# Using a pre-trained Transformer Encoder-Decoder based Paraphraser
 
import os.path
from os import path
import json
import torch
import nltk
import sentencepiece
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BertTokenizer

nltk.download('punkt')

# Using a standard model
tokenizer_model_name = 'google/pegasus-large'

# Using a pre-trained community model
paraphrasing_model_name = 'tuner007/pegasus_paraphrase'

torch_device = 'cuda'
if torch.cuda.is_available() == False:
  torch_device = 'cpu'

print('Device: ' + str(torch_device), flush = True)

tokenizer = PegasusTokenizer.from_pretrained(tokenizer_model_name)
model = PegasusForConditionalGeneration.from_pretrained(paraphrasing_model_name).to(torch_device)

def get_unit_paraphrase(input_text, num_return_sequences = 1, num_beams = 10):
  max_len = 60
  batch = tokenizer([input_text], 
                    truncation = True,
                    padding = 'longest',
                    max_length = max_len,
                    return_tensors = 'pt').to(torch_device)

  translated = model.generate(**batch,
                              max_length = max_len,
                              num_beams = num_beams,
                              num_return_sequences = num_return_sequences,
                              temperature = 1.5)
  
  target = tokenizer.batch_decode(translated,
                                  skip_special_tokens = True)
  
  return target[0]

def get_paraphrase(input_text, num_return_sequences = 2, num_beams = 10):
  preprocess_len = 52
  complete_paraphrase = ''

  sentences = nltk.sent_tokenize(input_text)
  for sentence in sentences:
    tokens = nltk.word_tokenize(sentence)
    count = len(tokens)
    if count > preprocess_len:
      continue
    try:
      sentence_paraphrase = get_unit_paraphrase(sentence)
    except:
      continue
    complete_paraphrase += sentence_paraphrase + ' '

  return complete_paraphrase

# Viewing Sample Paraphrases

examples = ['you should watch louis le vau \'s latest video . steven oh of tyt is disturbing as hell and makes me hope that jimmy dore wakes the left up .',
            'kill yourself you whiny , self-righteous faggot .',
            'but why do they make that face']

for example in examples:
  print('Source: ' + str(example))
  response = get_paraphrase(example)
  print('Primary Paraphrase: ' + str(response))
  print()

# Generating the Augmented Training Data
set_type = 'reddit'

# Reference to the absolute path in Google Drive
data_path = './' + set_type

with open(data_path + '/' + 'train.json', 'r+') as f:
  raw_train = json.load(f)

# Getting the Augmented Datapoints
augmented_data = []
overwrite_data = True
save_name = data_path + '/' + 'sent_wise_paraphrased_train.json'
limit = len(raw_train)
minimum_length = 4
interval = 500

test_mode = False
if test_mode == True:
  interval = 2
  limit = 50

backup_present = path.isfile(save_name)
done = 0

if backup_present:
  print('Pre-processed Data Backup Found: ' + str(backup_present), flush = True)
  with open(save_name, 'r+') as f:
    augmented_data = json.load(f)
  done = len(augmented_data)
  print('Starting from ' + str(done) + ' onwards.', flush = True)

for index in tqdm(range(done, limit)):
  unit = raw_train[index]
  if index > limit - 1:
    break

  try:
    raw_text = str(unit['source'].replace('\n', ' '))
    target = get_paraphrase(raw_text)
  except:
    pass
    continue
  
  token_count = len(nltk.word_tokenize(target))
  if token_count < minimum_length:
    continue
  new_unit = unit.copy()
  
  if 'type' in new_unit:
    new_unit.pop('type')
  if 'set' in new_unit:
    new_unit.pop('set')

  new_unit['source'] = target
  augmented_data.append(new_unit)

  if index % interval == 0 and overwrite_data == True:
      with open(save_name, 'w+') as f:
        json.dump(augmented_data, f)

if overwrite_data == True:
  with open(save_name, 'w+') as f:
    json.dump(augmented_data, f)

# ^_^ Thank You