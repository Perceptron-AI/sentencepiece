import re 
import collections


class bytepair(object):

    def get_vocab(self, filename):
        vocab = collections.defaultdict(int)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split()
                for word in words:
                    vocab[' '.join(list(word)) + ' </w>'] += 1
                break
        return vocab

        
    def bigram((self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            w = word.split()
            for i in range(len(w)-1):
                pairs[w[i],w[i+1]] += freq
        return pairs


    def merge_vocab((self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        bytepair = ''.join(pair)
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out, (bigram, bytepair)

    def find_merge(self, vocab, tokens, num_merge):
        merges = []
        for i in range(num_merges):
            pairs = bigram(vocab)
            if not pairs:
                break
            best_pairs = max(pairs, key=pairs.get)
            best_count = pairs[best_pairs]
            vocabs, (bigram, bytepair) = self.merge_vocab(best_pairs, vocab)
            merges.append((r'(?<!\S)' + bigram + r'(?!\S)', bytepair))
            tokens[bytepair] = best_pairs
        return vocab, tokens, merges

    
    @property
    def get_tokens((vocab):
        tokens = collections.defaultdict(int)
        for word, freq in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                tokens[token] += freq
        return tokens


# vocab = get_vocab('data/text.txt')
# pairs = get_stats(vocab)
# best = max(pairs, key=pairs.get)

# print('==========')
# print('Tokens Before BPE')
# tokens = get_tokens(vocab)
# print('Tokens: {}'.format(tokens))
# print('Number of tokens: {}'.format(len(tokens)))
# print('==========')

# num_merges = 1
# for i in range(num_merges):
#     pairs = get_stats(vocab)
#     if not pairs:
#         break
#     best = max(pairs, key=pairs.get)
#     vocab = merge_vocab(best, vocab)
#     print('Iter: {}'.format(i))
#     print('Best pair: {}'.format(best))
#     tokens = get_tokens(vocab)
#     print('Tokens: {}'.format(tokens))
#     print('Number of tokens: {}'.format(len(tokens)))
#     print('==========')
