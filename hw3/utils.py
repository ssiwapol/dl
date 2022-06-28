import Levenshtein
from phonemes import PHONEME_MAP


# Decode and calculate Levenshtein distance
def cal_levenshtein(h, y, lh, ly, decoder):

    # Beam search
    beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(h, seq_lens=lh)

    # Find batch size
    batch_size = h.shape[0]

    # Loop through each element
    dist = 0
    for i in range(batch_size):
        # Get the first beam result
        h_sliced = beam_results[i][0][:out_seq_len[i][0]]
        # Map using PHONEME_MAP
        h_string = ''.join([PHONEME_MAP[j] for j in h_sliced])

        # Extract y
        y_sliced = y[i][:ly[i]]
        # Map Using PHONEME_MAP
        y_string = ''.join([PHONEME_MAP[j] for j in y_sliced])
        
        # Calculate Levenshtein distance
        dist += Levenshtein.distance(h_string, y_string)

    # Calculate distance avg
    dist /= batch_size

    return dist

def decode_h(h, lh, decoder):
    # Beam search
    beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(h, seq_lens=lh)

    # Find batch size
    batch_size = h.shape[0]

    # Loop through each element
    y_pred = []
    for i in range(batch_size):
        # Get the first beam result
        h_sliced = beam_results[i][0][:out_seq_len[i][0]]
        # Map using PHONEME_MAP
        h_string = ''.join([PHONEME_MAP[j] for j in h_sliced])

        # Append output
        y_pred.append(h_string)
    
    return y_pred
