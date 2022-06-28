# import matplotlib.pyplot as plt
# import seaborn as sns

from letter import LETTER_LIST


def create_dictionaries(letter_list):
    '''
    Create dictionaries for letter2index and index2letter transformations
    based on LETTER_LIST

    Args:
        letter_list: LETTER_LIST

    Return:
        letter2index: Dictionary mapping from letters to indices
        index2letter: Dictionary mapping from indices to letters
    '''
    letter2index = dict()
    index2letter = dict()
    # TODO
    for i, l in enumerate(letter_list):
        letter2index[l] = i
        index2letter[i] = l
    return letter2index, index2letter
    

def transform_index_to_letter(batch_indices, letter_list=LETTER_LIST):
    '''
    Transforms numerical index input to string output by converting each index 
    to its corresponding letter from LETTER_LIST

    Args:
        batch_indices: List of indices from LETTER_LIST with the shape of (N, )
    
    Return:
        transcripts: List of converted string transcripts. This would be a list with a length of N
    '''
    transcripts = []
    # TODO
    for i in batch_indices:
        transcripts.append(letter_list[i])
    return transcripts
        
# Create the letter2index and index2letter dictionary
letter2index, index2letter = create_dictionaries(LETTER_LIST)


# def plot_attention(attention):
#     # utility function for debugging
#     plt.clf()
#     sns.heatmap(attention, cmap='GnBu')
#     plt.show()
