import numpy as np
import dataloader


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def beam_search_decoder(encoder_model, decoder_model, target_i2c, target_w2i, input_seq, decoding_json, k=10):
    """ Beam search decoder 
        reference from: https://github.com/mmehdig/lm_beam_search/blob/master/beam_search.py
    """
    try:
        decoder_model.get_layer(name='attention')
        has_attention = True
    except:
        has_attention = False

    D = dataloader.load_dict_from_json(decoding_json)
    #D = dataloader.load_dict_from_json('word_models/word_decoding.json')
    max_decoder_seq_length = D['max_decoder_seq_length'] - 2
    ##num_unique_target_chars = 4 #don't need this
    #states_value = encoder_model.predict(input_seq)
    vals = encoder_model.predict(input_seq)

    if has_attention:
        states_value = vals[1:]
        enc_outs = vals[0]
    else:
        states_value = vals

    stop_condition = False
    decoded_sentence = ''
    ##target_seq = np.zeros((1,1))
    ##target_seq[0,0] = char_ind
    cumlogprob = 0.0
    
    # (log(1), initialize_of_zeros)
    k_beam = [(cumlogprob, np.array([target_w2i['START_']] + [target_w2i['_END']] * max_decoder_seq_length).reshape(1, -1))]  
    ended = []

    # l : point on target sentence to predict
    for l in range(max_decoder_seq_length):
        all_k_beams = []

        for prob, sent_predict in k_beam:
            if has_attention:
                output_tokens = decoder_model.predict([sent_predict] + states_value + [enc_outs])[0]
            else:
                output_tokens = decoder_model.predict([sent_predict] + states_value)[0]
 
        # top k!
        possible_k = output_tokens[0, l, :].argsort()[-k:][::-1]
      
        # add to all possible candidates for k-beams
        all_k_beams += [
            (
                (sum(np.log(output_tokens[0, i, sent_predict[0,i+1]]) for i in range(l)) + np.log(output_tokens[0,l,next_wid])) / (l+1),
                np.array(list(sent_predict[0,:l+1]) + [next_wid] + [target_w2i['_END']] * (max_decoder_seq_length -l-1)).reshape(1,-1)
            )
            for next_wid in possible_k
        ]
        
        k_beam = sorted(all_k_beams, key=lambda x: (x[0], x[1]))[-k:]
        
        # check if there is sentence that has already ended
        to_pop = []
        for i in reversed(range(len(k_beam))):
            sent = k_beam[i][1]
            pred_token = sent[0,l+1]

            #if pred_token in [target_w2i['_END'], target_w2i['entity.n.01']]:
            if pred_token == target_w2i['_END']:
                ended.append(k_beam[i])
                del k_beam[i]
            
            elif pred_token in sent[0,:l+1]:
                sent[0,l+1] = target_w2i['_END']
                ended += [(k_beam[i][0], np.array(sent).reshape(1,-1))]
                del k_beam[i]

    k_beam = sorted(k_beam + ended, key=lambda x: (x[0], x[1]), reverse=True)[:k]

    #scores = softmax([beam[0] for beam in k_beam])
    k_sents = [' '.join([target_i2c[token] for token in beam[1][0][1:]]) for beam in k_beam]   
    
    return list(set(k_sents))

