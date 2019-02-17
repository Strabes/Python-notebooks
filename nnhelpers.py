###################################
## Neural Network Helper Functions
###################################

import numpy as np

def vectorizeTokenLists(input_tokens, output_tokens, vocab_dict,
                        input_token_length = None, output_token_length = None):
    """
    Create arrays of one-hot-encoded tokens
    
    Arguments:
    input_tokens: list of lists of tokens as inputs
    output_tokens: list of lists of tokens as outputs
    vocab_dict: dictionary mapping vocab terms to indices
    input_token_length: number of input tokens per observation
    output_token_length: number of output tokens per observation
    
    Returned:
    X: 3-dimensional array of encoded tokens. Array dimensions are:
        (number of observations) x (length of input token) x (tokens in vocabulary)
    y: 2-dimensional array of encoded target tokens. Array dimensions are:
        (number of observations) x ([length of output token] x [tokens in vocabulary])
    """
    VOCAB_SIZE = len(vocab_dict)
    if input_token_length is None:
        input_token_length = len(input_tokens[0])
    if output_token_length is None:
        output_token_length = len(output_tokens[0])
    #print('Vectorization...')
    X = np.zeros((len(input_tokens), input_token_length, VOCAB_SIZE), dtype=np.int)
    y = np.zeros((len(input_tokens), output_token_length * VOCAB_SIZE), dtype=np.int)
    for i, input_token in enumerate(input_tokens):
        for t, token in enumerate(input_token):
            if token in vocab_dict:
                X[i, t, vocab_dict[token]] = 1
        for j, token in enumerate(output_tokens[i]):
            if token in vocab_dict:
                y[i, j * VOCAB_SIZE + vocab_dict[token]] = 1
    return X, y
	
	
	
def createInputOutputTokenLists(texts, input_token_length = 3,
                               output_token_length = 2, step = 1):
    '''
    Create lists of input and output token lists
    
    Arguments:
    texts: list containing one element for each text comprised of sequential list of tokens
    input_token_length: number of tokens to use as input for each returned observation
    output_token_length: number of tokens to use as output for each returned observation
    step: number of positions to move along the tokenized text to create each returned observation
    
    Returned:
    input_tokens: list of lists of input_token_length tokens
    output_tokens: list of lists of output_token_length tokens
    '''
    input_tokens = []
    output_tokens = []
    for text in texts:
        for i in range(0, len(text) - input_token_length - output_token_length + 1, step):
            input_tokens.append(text[i: (i + input_token_length)])
            output_tokens.append(text[(i + input_token_length): (i + input_token_length + output_token_length)])
    return input_tokens, output_tokens
	
	
	
	
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)