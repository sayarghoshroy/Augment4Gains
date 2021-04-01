import os
import json
import torch
import argparse

from multiprocessing import cpu_count
from model import SentenceVAE
from utils import to_var, idx2word, interpolate
from hate_speech_data import HSDataset
from torch.utils.data import DataLoader

def main(args):
    dataset = HSDataset('train', args)


    model = SentenceVAE(
        vocab_size=len(dataset.get_w2i()),
        sos_idx=dataset.sos_idx,
        eos_idx=dataset.eos_idx,
        pad_idx=dataset.pad_idx,
        unk_idx=dataset.unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    if args.paraphrase:
        data_loader = DataLoader(
                    dataset=dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=cpu_count(),
                    pin_memory=torch.cuda.is_available()
                )

        for iteration, batch in enumerate(data_loader):
            batch_size = batch['input'].size(0)

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = to_var(v)

            # Forward pass
            logp, mean, logv, z = model(batch['input'], batch['length'])
            samples, _ = model.inference(z=z)
            with open(os.path.join(args.data_dir, args.augments_file), 'a') as f:
                i2w = dataset.get_i2w()
                w2i = dataset.get_w2i()      
                f.write('\n'.join(idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])) + '\n')
    else:
        samples, z = model.inference(n=args.num_samples)
        with open(os.path.join(args.data_dir, args.augments_file), 'a') as f:
            i2w = dataset.get_i2w()
            w2i = dataset.get_w2i()
            f.write('\n'.join(idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=int, nargs='+')
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)
    parser.add_argument('-pp', '--paraphrase', action='store_true')
    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-af', '--augments_file', type=str)
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-bs','--batch_size', type=int, default=32)
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
