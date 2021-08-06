from nltk.corpus import wordnet as wn
import re
import sys
import pandas as pd
from tqdm import tqdm

in_fname = sys.argv[1]  #$RESULTS.txt
out_fname = sys.argv[2] #$RESULTS.out.txt
is_WN = False

# ------------------------------- #
# NB. Output column order:
# ------------------------------- #
#   relation
#
#   node (hyponym)
#   node_lexname
#
#   gold (hypernym)
#   gold_lexname
#
#   pred (hypernym)
#   pred_lexname
#
#   is_gold_wn18rr_dev
#   is_gold_wordnet
#
#   lexname identity (wn18rr_dev)
#   lexname identity (wordnet)
#
#   Wu & Palmer similarity (wn18rr_dev)
#   Wu & Palmer similarity (wordnet)
# ------------------------------- #

def pred_gold_ident(pred_syn, gold_syn):
    """
    1. check if the predicted synset equals to the gold synsets in validation set
    2. check if the predicted synset is in the hypernyms set of wordNet
    """
    is_gold_dev = pred_syn == gold_syn  # needs revision (cases where multiple golds exist in WN18RR's train & dev set)
    is_gold_wn = pred_syn in node_syn.hypernyms() or pred_syn in node_syn.instance_hypernyms()
    return is_gold_dev, is_gold_wn


def lex_ident(node_syn, pred_syn, gold_syn):
    """
    1. check if the predicted synset's lexname equals to the gold synset's name
    2. check if the predicted synset's lexname is in the set of its wordnet hypernyms' lexnames (including hypernyms/instance_hypernyms)
    """
    pred_lex = pred_syn.lexname()
    gold_lex = gold_syn.lexname()

    lex_ident_dev = pred_lex == gold_lex
    lex_ident_wn = pred_lex in [x.lexname() for x in node_syn.instance_hypernyms()] \
                   or pred_lex in [x.lexname() for x in node_syn.hypernyms()]
    return lex_ident_dev, lex_ident_wn, pred_lex, gold_lex


def wup_score(pred_syn, gold_syn):
    """ Calculate the wup score for predicted synset and gold synset"""
    if pred_syn == gold_syn:
        wup_dev = 1.00
    else:
        wup_dev = pred_syn.wup_similarity(gold_syn)
        if wup_dev is None:
            wup_dev = 0.00
    return wup_dev


def nb_pred_in_gold(pred_syns, gold_syns):
    cnt = 0
    for pred_syn in pred_syns:
        if pred_syn in gold_syns:
            cnt += 1
    return cnt 


def mean_max_wup_score(pred_syns, gold_syns, is_wn = True):
    if is_wn:
        res = []
        for pred in pred_syns:
            wup_wn = [wup_score(pred, hyper) for hyper in gold_syns]
            if len(wup_wn) == 0 or all(score is None for score in wup_wn):
                wup_wn_max = 0.00
            else:
                wup_wn_max = max(wup_wn)
            res.append(wup_wn_max)
        return sum(res) / len(res)
    else:
        return 0.00


def hits_at_k(pred_labels, gold_labels, k):
    top_k = pred_labels[:k]
    res = sum([gold in top_k for gold in gold_labels]) / len(gold_labels)
    if len(gold_labels) <= k:
        return res
    else:
        return res * (len(gold_labels)/k)



 

            

if is_WN:
    with open(in_fname, 'r') as f:
        corpus = []
        for line in f:
            line = re.sub('\n', '', line)
            if line != u'':
                corpus.append(line)

    with open(out_fname, 'w') as rerank_file:
        rerank_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n' \
                          .format('relation', 'node', 'gold_syns', 'pred_k',
                                  'hits@1','hits@3','hits@10', 'mean_max_wup'))

        for line in tqdm(corpus): # each predicted pairs 
            node, rel_raw, gold, pred_k = line.split('\t')
            pred_k = pred_k.split('|')
    
            rel_raw = rel_raw[1:]
    
            # Load node and gold synsets from WordNet in NLTK
            node_syn = wn.synset(node)
            gold_syns = node_syn.hypernyms() + node_syn.instance_hypernyms()
    
            # If the line has no prediction at all: pred = '<unk>'
            #for i in range(len(pred)):
            for i in range(len(pred_k)):
                if pred_k[i] == '':
                    pred_k[i] = '<unk>'

            # Relation type definition
            if rel_raw == 'hypernym':
                if '.v.' in node:
                    rel = rel_raw + '_v'
                else:  # '.n.' in node:
                    rel = rel_raw + '_n'
            else:  # 'instance_hypernym'
                rel = rel_raw

            pred_syns = [wn.synset(pred) for pred in pred_k if pred != '<unk>'] 
            mean_max_wup = mean_max_wup_score(pred_syns, gold_syns, is_wn = True)
            hits_at_1 = hits_at_k(pred_syns, gold_syns, 1)
            hits_at_3 = hits_at_k(pred_syns, gold_syns, 3)
            hits_at_10 = hits_at_k(pred_syns, gold_syns, 10)

            gold_syns = [syn.name() for syn in gold_syns]

            rerank_file.write('{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.5f}\n' \
                              .format(rel, node, '|'.join(gold_syns), '|'.join(pred_k), hits_at_1, hits_at_3, hits_at_10, mean_max_wup))
       
else:
    corpus = pd.read_csv(in_fname, sep='\t', names=["node", "relation", "gold_syns", "pred_k"])

    with open(out_fname, 'w') as rerank_file:
        rerank_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n' \
                          .format('relation', 'node', 'gold_syns', 'pred_k',
                                  'hits@1','hits@3','hits@10', 'mean_max_wup'))

        for index, line in tqdm(corpus.iterrows()): 
            node = line.node
            rel = line.relation[1:]
            gold_syns = list(set(corpus[corpus['node'] == node]['gold_syns'].tolist()))
            pred_syns = line.pred_k.split('|')

            mean_max_wup = mean_max_wup_score(pred_syns, gold_syns, is_wn = False)
            hits_at_1 = hits_at_k(pred_syns, gold_syns, 1)
            hits_at_3 = hits_at_k(pred_syns, gold_syns, 3)
            hits_at_10 = hits_at_k(pred_syns, gold_syns, 10)

            rerank_file.write('{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.5f}\n' \
                            .format(rel, node, '|'.join(gold_syns), '|'.join(pred_syns), hits_at_1, hits_at_3, hits_at_10, mean_max_wup))

