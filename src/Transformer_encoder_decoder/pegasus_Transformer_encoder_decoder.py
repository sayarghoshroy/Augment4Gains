# Generating Samples using a Transformer Encoder-Decoder based Paraphraser

import json
import torch
import sentencepiece
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BertTokenizer

# Using a standard model
tokenizer_model_name = 'google/pegasus-large'

# Using a pre-trained community model
paraphrasing_model_name = 'tuner007/pegasus_paraphrase'

torch_device = 'cuda'
if torch.cuda.is_available() == False:
  torch_device = 'cpu'

tokenizer = PegasusTokenizer.from_pretrained(tokenizer_model_name)
model = PegasusForConditionalGeneration.from_pretrained(paraphrasing_model_name).to(torch_device)

def get_paraphrase(input_text, num_return_sequences = 2, num_beams = 10):
  batch = tokenizer([input_text], 
                    truncation = True,
                    padding = 'longest',
                    max_length = 128,
                    return_tensors = 'pt').to(torch_device)

  translated = model.generate(**batch,
                              max_length = 128,
                              num_beams = num_beams,
                              num_return_sequences = num_return_sequences,
                              temperature = 1.5)
  
  target = tokenizer.batch_decode(translated,
                                  skip_special_tokens = True)
  
  return target

# Viewing Sample Paraphrases

examples = ['you should watch louis le vau \'s latest video . steven oh of tyt is disturbing as hell and makes me hope that jimmy dore wakes the left up .',
            'kill yourself you whiny , self-righteous faggot .',
            'but why do they make that face']

for example in examples:
  print('Source: ' + str(example))
  responses = get_paraphrase(example)
  print('Primary Paraphrase: ' + str(responses[0]))
  print()

# Generating the Augmented Training Data
set_type = 'reddit'

data_path = './data/' + set_type

with open(data_path + '/' + 'train.json', 'r+') as f:
  raw_train = json.load(f)

# Getting the Augmented Datapoints
augmented_data = []
limit = len(raw_train)

test_mode = False
if test_mode == True:
  limit = 10

for index, unit in enumerate(tqdm(raw_train, total = limit)):
  if index > limit - 1:
    break
  raw_text = unit['source']
  target = get_paraphrase(example)[0]
  new_unit = unit

  new_unit['source'] = target
  augmented_data.append(new_unit)

with open(data_path + '/' + 'paraphrased_train.json', 'w+') as f:
  json.dump(augmented_data, f)

# ^_^ Thank You