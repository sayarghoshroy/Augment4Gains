import json
import random

def make_set(label, line, set_type):
    object_set = {}
    object_set['set'] = set_type
    object_set['main_index'] = random.randint(1000,9999)
    object_set['index'] = random.randint(1000,9999)
    object_set['type'] = 'sentences'
    object_set['source'] = line    
    object_set['target'] = label
    return object_set

aug_dat = []
counter = 0

def call_file(filename, label, set_type):
    with open(filename) as fp:
        lines = fp.readlines()
        for line in lines:
            aug_set= make_set(label, line.strip(), set_type)
            aug_dat.append(aug_set)
            global counter
            counter += 1

call_file('augment_gab0.txt', 0, 'gab')
call_file('augment_gab1.txt', 1, 'gab')

print(len(aug_dat))
print(aug_dat[-1])
jstr = json.dumps(aug_dat, ensure_ascii=False, indent=4)

with open('genaug_gab.json', 'w+') as outfile:
    json.dump(aug_dat, outfile)
