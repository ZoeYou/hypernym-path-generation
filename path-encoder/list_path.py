from nltk.corpus import wordnet as wn
from tqdm import tqdm

## create paths for a list of synsets
with open('../WN18RR-hp/noun/train.txt') as in_f:
    with open('all_paths.txt', 'w') as out_f:
        max_len = 0
        for line in tqdm(in_f.readlines()):
            l = line.split('\t')[0]
            paths = wn.synset(l.strip()).hypernym_paths()
            '''
            try:
                assert len(paths) == 1
            except(AssertionError):
                print(len(paths))
                print(paths)
                raise(AssertionError)
            '''
            assert type(paths) == list
            for path in paths:
                max_len = max(max_len, len(path))
                for hypernym in path:
                    out_f.write(hypernym.name())
                    out_f.write('\t')
                out_f.write('\n')
    print(max_len)
