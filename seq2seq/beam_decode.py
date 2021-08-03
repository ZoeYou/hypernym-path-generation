import numpy as np
import dataloader

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
                output_tokens, h, c = decoder_model.predict([sent_predict] + states_value + [enc_outs]) #TODO
            else:
                output_tokens, h, c = decoder_model.predict([sent_predict] + states_value)

        states_value = [h,c]
        # top k!
        possible_k = output_tokens[0, l, :].argsort()[-k:][::-1]
        print('possible k', possible_k)
      
        # add to all possible candidates for k-beams
        all_k_beams += [
            (
                sum(np.log(output_tokens[0, i, sent_predict[0,i+1]]) for i in range(l)) + np.log(output_tokens[0,l,next_wid]),
                np.array(list(sent_predict[0,:l+1]) + [next_wid] + [target_w2i['_END']] * (max_decoder_seq_length -l-1)).reshape(1,-1)
            )
            for next_wid in possible_k
        ]
        # top 2k (not top k, just in case all the k sentences are ended)
        k_beam = sorted(all_k_beams)[-2*k:]
        print(66666)
        print(k_beam)



        
        # check if there is sentence that has already ended
        to_pop = []
        for i in reversed(range(len(k_beam))):
            sent = k_beam[i][1]
            if sent[0, l+1] == target_w2i['_END']:
                ended.append(k_beam[i])
                k_beam.pop(i)

        k_beam = k_beam[:k]
  
    k_beam = sorted(k_beam + ended, reverse=True)[:k]
    k_sents = [' '.join([target_i2c[token] for token in beam[1][0][1:]]) for beam in k_beam]    
    return k_sents

