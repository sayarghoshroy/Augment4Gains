# Meanings of various flags

- `name`: name of the experiment
- `path`: path to directory where the data files are stored: the directory needs to have `train.json`, `test.json`, and `val.json`
- `test_mode`: `1` to test out the pipeline quickly on a small portion of the data, `0` otherwise
- `use_aug`: whether to apply augmentation 
- `aug_src`: name of the `json` file having the augmented examples ~ As an example: 'tf_paraphrase.json'
- `model`: `0` for BERT, `1` for RoBERTa
- `bal`: `0` for no class imbalance corrections, `1` for the first method, `2` for the second method

---