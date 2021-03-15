import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers import AutoTokenizer

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

def augment(model, tokenizer, lines, n_augments=5, temperature=1, mask_token='[MASK]', device="cpu"):
    tokens = tokenizer(lines, return_tensors="pt", padding=True, truncation=True)
    tokens.to(device)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("source_data", type=str)
    parser.add_argument("dest_prefix", type=str)
    parser.add_argument("n_augment", type=int)
    parser.add_argument("pretrained_model_or_path", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model_or_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_or_path)

    with open(args.source_data, 'r') as f:
        source_data = json.load(f)
    
    augmented_data = [list() for j in range(args.n_augment)]
    i = 0
    while i < len(source_data):
        batch = [x["source"] for x in source_data[i:i + args.batch_size]]
        masked_samples = simple_mask(batch)
        augments = augment(model, tokenizer, masked_samples, n_augments=args.n_augment, temperature=args.temperature, device=device)
        for j in range(args.n_augment):
            augmented_data[j].extend(source_data[i:i + args.batch_size])
            for k in range(i, i + len(batch)):
                augmented_data[j][k]["source"] = augments[j][k - i]

        i += args.batch_size
        print("{} out of {} samples processed".format(i, len(source_data)))
    
    for j in range(args.n_augment):
        with open("{}_{}.json".format(args.dest_prefix, j), 'w') as f:
            json.dump(augmented_data[j], f)