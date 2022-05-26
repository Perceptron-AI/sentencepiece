import numpy as np 
import sentencepiece 
import collection 
from tqdm import tqdm
from pathlib import Path



UNK = '<unk>'
END_OF_LINE = '<endofline>'
END_OF_TEXT = '<endoftext>'
WORD_START = '▁'


def SentencePiece_train(PATH):
    sp_text = Path(PATH)
    if sp_text.exists():
        print(f'Using existing "{sp_text}", remove and re-run if it is stale.')

    spm.SentencePieceTrainer.train(input=sp_text, model_prefix='m', vocab_size=10000, 
                                    model_type='bpe', max_sentence_length=16384, bos_id=-1,
                                    eos_id=-1, unk_piece='UNK')


   