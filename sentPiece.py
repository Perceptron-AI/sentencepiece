import numpy as np 
import sentencepiece as spm
import collections 
from tqdm import tqdm
from pathlib import Path



UNK = '<unk>'
END_OF_LINE = '<endofline>'
END_OF_TEXT = '<endoftext>'
WORD_START = '▁'


def trainer(PATH, output_path='.'):
    sp_text = Path(PATH)
    if sp_text.exists():
        print(f'Using existing "{sp_text}", remove and re-run if it is stale.')

    spm.SentencePieceTrainer.train(input=sp_text, model_prefix='m', vocab_size=10000, 
                                    model_type='bpe', max_sentence_length=16384, bos_id=-1,
                                    eos_id=-1, unk_piece='UNK')

    sp_model = spm.SentencePieceProcessor()
    sp_model = spm.SentencePieceProcessor()
    assert sp_model.load('m.model')
    eot = sp_model.PieceToId(END_OF_TEXT)
    eol = sp_model.PieceToId(END_OF_LINE)
    dtype = np.uint16 if len(sp_model) < 2**16 - 1 else np.uint32

    encoded_splits = collections.defaultdict(list)
    encoded = []

    def append_and_clear(x):
        encoded_splits['train'].append(np.array(x, dtype=dtype))
        x.clear()

    with open(sp_text, 'r') as f:
        for line in f:
            encoded.extend(sp_model.EncodeAsIds(line))
            encoded.append(eol)
            if len(encoded) > 100000:
                append_and_clear(encoded)
            encoded.append(eot)
        append_and_clear(encoded)

    output_root = Path(output_path)

    split_path = output_root / 'output.npy'
    print(f'Saving encoded split to {split_path}')
    encoded = np.concatenate(encoded_splits['train'])
    assert encoded.dtype == dtype
    np.save(split_path, encoded)

if __name__ == '__main__':
    print('s')
    trainer('data/text.txt')
    print('done')
   