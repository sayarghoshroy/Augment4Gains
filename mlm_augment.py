from transformers import AutoConfig, AutoModelForMaskedLM
from transformers import AutoTokenizer
# from transformers.data.data_collator import DataCollatorForWholeWordMask
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def simple_mask(lines, mask_token='[MASK]', masking_probability=0.15):
    tokenized = [line.split() for line in lines]
    max_len = max([len(line) for line in tokenized])
    probability_matrix = torch.full((len(tokenized), max_len), masking_probability)
    for i, line in enumerate(tokenized):
        probability_matrix[i, len(line):] = 0
    masked_indices = torch.bernoulli(probability_matrix).bool()
    for i in range(masked_indices.shape[0]):
        for j in range(masked_indices.shape[1]):
            if masked_indices[i, j]:
                tokenized[i][j] = mask_token
    
    return [' '.join(line) for line in tokenized]

def augment(model, tokenizer, lines, n_augments=5, temperature=1, mask_token='[MASK]'):
    tokens = tokenizer(lines, return_tensors="pt", padding=True, truncation=True)
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    masked_indices = torch.where((tokens.input_ids == mask_token_id), torch.ones(tokens.input_ids.shape), torch.zeros(tokens.input_ids.shape)).bool()

    outputs = model(**tokens, return_dict=True)
    probs = F.softmax((1 / temperature) * outputs.logits, dim=2)
    samples = torch.multinomial(probs[masked_indices], n_augments)

    new_samples = list()
    for i in range(n_augments):
        augmented = tokens.input_ids.clone()
        augmented[masked_indices] = samples[:, i]
        new_samples.append([tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(row, skip_special_tokens=True)) for row in augmented.detach().tolist()])
    
    return new_samples