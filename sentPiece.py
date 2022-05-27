import numpy as np 
import sentencepiece as spm
import collections 
from tqdm import tqdm
from pathlib import Path
from glob import glob

UNK = '<unk>'
END_OF_LINE = '<endofline>'
END_OF_TEXT = '<endoftext>'
WORD_START = '▁'

class data_preprocessing(object):
    def __init__(self, 
                sub_dim, 
                training_file, 
                output_path:Path, 
                prefix="subwords", 
                model_type="bpe",
                max_sentence_length=100000,
                vocab_size = 10000,
                ):
        self.prefix = prefix + "_" + str(sub_dim) + "_" + model_type
        self.training_file = training_file
        self.output_path = output_path

        spm.SentencePieceTrainer.train(input=training_file,     
                                    model_prefix=self.prefix, 
                                    vocab_size=vocab_size, 
                                    model_type=model_type, 
                                    max_sentence_length=max_sentence_length, 
                                    bos_id=-1,
                                    eos_id=-1, 
                                    unk_piece='UNK')

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.prefix+".model")

    def encoder(self, text):
        ids = self.sp.EncodeAsIds(text)
        return ids 

    def dencoder(self, id_array):
        text = self.sp.self.sp.decode_ids(id_array)
        return text 

    def save_as_npy(self):
        eot = self.sp.PieceToId(END_OF_TEXT)
        eol = self.sp.PieceToId(END_OF_LINE)
        dtype = np.uint16 if len(self.sp) < 2**16 - 1 else np.uint32

        encoded_splits = collections.defaultdict(list)
        encoded = []

        def append_and_clear(x):
            encoded_splits['train'].append(np.array(x, dtype=dtype))
            x.clear()

        with open(self.training_file, 'r') as f:
            for line in f:
                encoded.extend(self.sp.EncodeAsIds(line))
                encoded.append(eol)
                if len(encoded) > 100000:
                    append_and_clear(encoded)
                encoded.append(eot)
            append_and_clear(encoded)

        output_root = Path(self.output_path)

        split_path = output_root / (self.prefix + ".npy")
        print(f'Saving encoded split to {split_path}')
        encoded = np.concatenate(encoded_splits['train'])
        assert encoded.dtype == dtype
        np.save(split_path, encoded)

``
# class GPT2Dataset(Dataset):
#     def __init__(self, corpus_path, max_len, n_tokens, dataset_plk_path="data/gpt2_dataset.npy"):

#         self.max_len = max_len
#         if os.path.isfile(dataset_plk_path):
#             self.corpus = np.load(dataset_plk_path, allow_pickle=True)
#         else:
#             self.corpus = []
#             processor = TextProcessor(n_tokens, corpus_path)
#             print("Encoding texts...")
#             with open(corpus_path, 'r') as file:
#                 for line in tqdm(file):
#                     encoded = processor.encode(line)
#                     self.corpus.append(np.array(encoded, dtype=np.int16))
#             gc.collect()
#             np.save(dataset_plk_path, self.corpus)

#     def __getitem__(self, index):
#         if index > self.__len__():
#             print(index)
#             raise IndexError()

#         encoded = self.corpus[index]
#         offset = random.randint(0, max(0, len(encoded) - self.max_len))
#         encoded = encoded[offset:self.max_len + offset]

#         return torch.LongTensor(encoded)

#     def __len__(self):
#         return len(self.corpus)


# def collate(batch):
#     return pad_sequence(batch, batch_first=True, padding_value=0)


if __name__ == "__main__":
    process = data_preprocessing(10, 'data/text.txt', output_path='.')
    process.save_as_npy()
