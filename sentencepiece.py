import re
import os
import collections
import numpy as np
from scipy.special import digamma

class Tree:
    def __init__(self):
        self.root = {}

    def add(self, word, value):
        node = self.root
        for ch in word:
            if ch not in node:
                node[ch] = node
            node = node[ch]
        node['<END>'] = value

    def get_value(self, word):
        node = self.root
        for ch in word:
            if ch not in node:
                return 0
            node = node[ch]
        if '<END>' not in node:
            return 0
        return node['<END>']

    def set_value(self, word, value):
        node = self.root
        for ch in word:
            if ch not in node:
                raise ValueError("word not in tree")
            node = node[ch]
        if '<END>' not in node:
            raise ValueError("word not in tree")
        node['<END>'] = value


class SentencePieceTrainer:
    def __init__(self):
        self.tree = None
        self.maxlen = None
        self.vocab_size = None

    def _initialize_tree(self, tokens):
        tree = Tree()
        norm = sum(list(tokens.values()))
        logsum = digamma(norm)

        maxlen = 0
        for tok, val in tokens.items():
            tree.add(tok, digamma(val)-logsum)
            maxlen = max(maxlen, len(tok))

        return tree, maxlen

    def forward_step(self, text, tree):
        N = len(text)

        # d[i] contains the maximum log_prob of any tokenization
        # of text[:i], initialized to 0 (i.e. log(0)=-infty)
        d = [-np.inf]*(N+1)

        # p[i] (stands for parent) contains the number of characters of
        # the final token in the most likely sequence that ends at index i
        p = [None]*(N+1)
        d[0]=0

        for i in range(1, N+1):

            # find all possible final words. Have to look back
            # a distance set by the length of the longest token
            for j in range(max(i-self.maxlen, 0), i):

                final_token = text[j:i]
                final_value = tree.get_value(final_token)

                # if the current ending word has a higher log-probability,
                # save that value and store the word (i.e. # chars to backtrack)
                if final_value and d[j]+final_value > d[i]:
                    d[i] = d[j]+final_value
                    p[i] = len(final_token)
            if p[i] is None:
                raise ValueError(f"Encountered unknown token '{text[i-1]}'.")

        loss = d[-1]
        return loss, p

    def backward_step(self, text, p):
        idx = len(p)
        tokenization = []
        while idx > 1:
            # move back the number of steps p tells you to
            next_idx = idx-p[idx-1]

            # extract the final token
            tok = text[next_idx-1:idx-1]
            tokenization.append(tok)

            idx = next_idx
        tokenization = list(reversed(tokenization))
        return tokenization

    def E_step(self, tokenization, tree):
        # get the new token counts based on updated tokenization
        counts = collections.Counter(tokenization)
        norm = sum(list(counts.values()))

        # Bayesianify them: https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
        # https://github.com/google/sentencepiece/blob/master/src/unigram_model_trainer.cc
        # we are returning the log probabilties here (alpha=0 prior)
        logsum = digamma(norm)
        for k, v in counts.items():
            counts[k] = digamma(v)-logsum

        for k, v in counts.items():
            tree.set_value(k, v)
        return tree

    def M_step(self, text, tree):
        loss, p = self.forward_step(text, tree)
        tokenization = self.backward_step(text, p)
        return tokenization, loss

    def EM_step(self, text, tokenization, tree):
        tree = self.E_step(tokenization, tree)
        tokenization, loss = self.M_step(text, tree)
        return loss, tokenization, tree

    def EM_round(self, text, tokens, delta=0.01, max_iter=10):
        tokenization, old_loss = self.M_step(text, self.tree)
        for step in range(max_iter):
            print(f"EM iter {step}: ", end='')
            loss, tokenization, tree = self.EM_step(text, tokenization, self.tree)
            print(f"Loss={loss:.2f}")
            if abs(old_loss-loss) < delta:
                break
            old_loss = loss

    def prune_tokens(self, tokens, characters, vocab_size, trim_frac=0.2):
        """ Tokens are passed by reference and modified in place.
        Returns:
            True: to indicate to caller that more rounds are needed
            False: to indicate we successfully hit the target vocab size
            ValueError: if the vocab size cannot be reached."""
        sorted_tokens = tokens.most_common()
        N = len(sorted_tokens)
        n_trim = int(trim_frac*N)
        for i in reversed(range(N)):
            if N <= vocab_size:
                return False
            if n_trim <= 0:
                return True
            tok = sorted_tokens[i][0]
            if tok not in characters:
                self.tree.set_value(tok, 0) # we need to delete it from the tree (that sticks around)
                tokens.pop(tok) # also need to delete from tokens, so the next round doesn't see it
                n_trim -= 1
                N -= 1
        if n_trim > 0:
            raise ValueError('Could not reduce tokens further. Please increase vocab size')
        return False

    def fit(self, text, tokens, characters, vocab_size, delta=0.01, max_iter=5, max_rounds=5):
        """ To turn off pruning, just set max_rounds=1 """
        text = re.sub(' ', '_', text)
        if vocab_size > len(tokens):
            raise ValueError(f"Vocab size is larger than the availble number of tokens {len(tokens)}.")
        self.tree, self.maxlen = self._initialize_trie(tokens)
        for i in range(1, max_rounds+1):
            print(f"--- Round {i}. Vocab size: {len(tokens)} ---")
            self.EM_round(text, tokens, delta, max_iter)
            if not self.prune_tokens(tokens, characters, vocab_size):
                break
        self.vocab_size = len(tokens)



    def generalized_forward_step(self, text, tree, nbest_size=1):
        N = len(text)
        d = [-np.inf]*(N+1)
        p = [None]*(N+1)
        d[0]=0
        for i in range(1, N+1):
            d_queue = []
            p_queue = []
            for j in range(max(i-self.maxlen, 0), i):
                final_token = text[j:i]
                final_value = tree.get_value(final_token)
                if final_value:
                    curr_d = d[j]+final_value
                    curr_p = len(final_token)
                    d[i] = max(d[i], curr_d)
                    d_queue.append(curr_d)
                    p_queue.append(curr_p)
            ids = np.argsort(d_queue)[-nbest_size:]
            p[i] = [p_queue[z] for z in ids]
        return p

    def generalized_backward_step(self, text, p):
        idx = len(p)
        tokenization = []
        while idx > 1:
            back_steps = np.random.choice(p[idx-1])
            next_idx = idx-back_steps
            tok = text[next_idx-1:idx-1]
            tokenization.append(tok)
            idx = next_idx
        tokenization = list(reversed(tokenization))
        return tokenization

    def tokenize(self, text, nbest_size=1):
        text = re.sub(' ', '_', text)
        if self.tree is None:
            raise ValueError("Trainer has not yet been fit. Cannot tokenize.")
        p = self.generalized_forward_step(text, self.tree, nbest_size)
        tokenization = self.generalized_backward_step(text, p)
        return tokenization