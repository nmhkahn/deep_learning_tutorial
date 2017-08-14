"""
This code is simplified version of 
https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/utils.py
"""

import collections 
import numpy as np
    
# this is for word count, this has to change to process efficiently with char-level
def build_vocab(sentences):
    word_counts = collections.Counter(sentences)
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv.append("<START>")
    vocabulary_inv.append("<END>")

    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

class TextLoader:
    def __init__(self, path):
        with open(path, "r") as _file:
            text = list(_file.read())
            
        self.vocab, self.words = build_vocab(text)
        self.text = np.array([self.vocab[word] for word in text])

        self.X = np.empty((len(self.text)+1), dtype=np.int64)
        self.y = np.empty((len(self.text)+1), dtype=np.int64)

        self.X[0] = self.vocab["<START>"]
        self.X[1:] = np.copy(self.text)
        
        self.y[:-1] = self.X[1:]
        self.y[-1]  = self.vocab["<END>"]

    def next_batch(self, batch_size, seq_length):
        start = np.random.randint(0, len(self.X)-batch_size*seq_length)
        end   = start + batch_size*seq_length

        X_to_return = self.X[start:end].reshape(batch_size, seq_length)
        y_to_return = self.y[start:end].reshape(batch_size, seq_length)
        
        return X_to_return, y_to_return
