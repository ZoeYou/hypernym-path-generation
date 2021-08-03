import pandas as pd
import sys

model_name = sys.argv[1]
pred_path = sys.argv[2]
score_path = sys.argv[3]


def h_at_1_wordnet(df_sub):
    ig_wn_hits_1_sum = sum(df_sub.loc[:, 'hits@1'])
    h_at_1_wn = ig_wn_hits_1_sum / len(df_sub)
    return round(h_at_1_wn, 4)

def h_at_3_wordnet(df_sub):
    ig_wn_hits_3_sum = sum(df_sub.loc[:, 'hits@3'])
    h_at_3_wn = ig_wn_hits_3_sum / len(df_sub)
    return round(h_at_3_wn, 4)

def h_at_10_wordnet(df_sub):
    ig_wn_hits_10_sum = sum(df_sub.loc[:, 'hits@10'])
    h_at_10_wn = ig_wn_hits_10_sum / len(df_sub)
    return round(h_at_10_wn, 4)

def wu_and_palmer_wordnet_max_avg(df_sub):
    wup_wn_max = sum(df_sub.loc[:, 'mean_max_wup']) / len(df_sub)
    return round(wup_wn_max, 4)


# Read tab-separated prediction file as pandas dataframe
df = pd.read_csv(pred_path, sep='\t')
rels = df.relation.unique()

with open(score_path, 'w') as f:
    f.write("model name: {}\n".format(model_name))
    f.write("model prediction: {}\n\n".format(pred_path))
    for relation_i in rels:
        f.write("relation: {}\n".format(relation_i))

        df_i = df.loc[df.loc[:,"relation"]==relation_i, :]

        h1_wn = h_at_1_wordnet(df_i)
        h3_wn = h_at_3_wordnet(df_i)
        h10_wn = h_at_10_wordnet(df_i)
        wup_wn_max_avg = wu_and_palmer_wordnet_max_avg(df_i)

        f.write("h@1\th@3\th@10\tmean_max_wup_wn\n")
        f.write("{}\t{}\t{}\t{}\n\n".format(h1_wn, h3_wn, h10_wn, wup_wn_max_avg))
                                        
