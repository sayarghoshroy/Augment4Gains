# WordNet based synonym substitution with WSD

- We first identify the words to replace. 
- Once we have identified the word, we then identify the sense of the word through WSD
- We then generate synonyms and hypernyms for the word  and replace it in place. 

## Instructions 

- Use `wordnet-helper.ipynb` to generate the sentences for which we need to perform WSD
- Then clone the BERT-WSD available [here](https://github.com/BPYap/BERT-WSD)
- Copy the script `generateSyns.py` to the `scripts/` folder in the cloned repo
- Run the script to generate the data based on the instructions given in the repo. 