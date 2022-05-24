import re 
import collections

class encoder(object):
    def __init__(self):
        self.vocab = None
        self.tokens = None
        self.merges = None

    def get_vocab(self, filename):
        vocab = collections.defaultdict(int)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    vocab[''.join(list(word)) + ' </w>'] += 1
        return vocab
        
    def bigram_count(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            w = word.split()
            for i in range(len(w)-1):
                pairs[w[i],w[i+1]] += freq
        return pairs


    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        bytepair = ''.join(pair)
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out, (bigram, bytepair)

    def get_tokens(self, vocab):
        tokens = collections.defaultdict(int)
        for word, freq in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                tokens[token] += freq
        return tokens

    def find_merge(self, vocab, num_merges):
        merges = []
        
        for i in range(num_merges):
            pairs = self.bigram_count(vocab)
            if not pairs:
                break
            best_pairs = max(pairs, key=pairs.get)
            best_count = pairs[best_pairs]
            vocab, (bigram, bytepair) = self.merge_vocab(best_pairs, vocab)
            merges.append((r'(?<!\S)' + bigram + r'(?!\S)', bytepair))
            tokens =  self.get_tokens(vocab)
        return vocab, merges, tokens

    def fit_train(self, filename, num_merges):
        vocab = self.get_vocab(filename)
        self.vocab, self.merges, self.tokens = self.find_merge(vocab, num_merges)


# Test
# if __name__ == "__main__":
#     e = encoder()
#     e.fit_train('data/text.txt', 2)
#     print(e.vocab)

