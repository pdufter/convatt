import numpy as np
import random
import re

from tqdm import tqdm
from collections import Counter, defaultdict
import pyconll

from utils.utils import *


class UD(object):
    def __init__(self, config):
        self.config = config
        self.sample_weights = None

    def merge_filter_tags(self, merge_dict):
        for part in self.X_raw:
            for i in range(len(self.X_raw[part])):
                self.X_raw[part][i] = np.array([[x[0], merge_dict[x[1]]] for x in self.X_raw[part][i]])
                self.X_raw[part][i] = np.array([x for x in self.X_raw[part][i] if x[1] != ""])

    def get_vocabulary(self):
        vocab = []
        labels = []
        if self.config['build_voc_on_all']:
            for_voc = self.X_raw['trn'].tolist() + self.X_raw['dev'].tolist() + self.X_raw['tst'].tolist()
        else:
            for_voc = self.X_raw['trn'].tolist()
        for s in for_voc:
            vocab.extend([x[0] for x in s])
            labels.extend([x[1] for x in s])
        vocab = Counter(vocab)
        self.n_vocab_orig = len(vocab)
        if self.config['num_voc'] == 0:
            # take half
            self.config['num_voc'] = self.n_vocab_orig // 2
        if self.config['num_voc'] != -1:
            vocab = vocab.most_common(n=self.config['num_voc'])
        else:
            vocab = vocab.most_common()
        labels = Counter(labels).most_common()
        print("n dictionary:", len(vocab))
        print("n labels:", labels)

        vocab = [x[0] for x in vocab]
        if self.config['start_end']:
            vocab.insert(0, self.config['END_token'])
            vocab.insert(0, self.config['START_token'])

        vocab.insert(0, self.config['OOV_token'])
        vocab.insert(0, self.config['PAD_token'])

        labels = [x[0] for x in labels]
        if self.config['start_end']:
            labels.insert(0, self.config['END_token'])
            labels.insert(0, self.config['START_token'])
        labels.insert(0, self.config['OOV_token'])
        labels.insert(0, self.config['PAD_token'])

        self.index2word = dict(enumerate(vocab))
        self.index2label = dict(enumerate(labels))

        self.word2index = defaultdict(lambda: 1, invdict(self.index2word))
        self.label2index = defaultdict(lambda: 1, invdict(self.index2label))
        # todo possibly use universal tagset

        self.vocab = set(vocab)

        self.n_labels = len(self.label2index)
        self.n_vocab = len(self.word2index)

    def transform_indices(self):
        self.X = {}
        self.Y = {}

        self.lengths = []

        for i, k in enumerate(["trn", "dev", "tst"]):
            self.X[k] = []
            self.Y[k] = []
            for sentence in self.X_raw[k]:
                sentence = sentence.tolist()
                self.lengths.append(len(sentence))

                if self.config['start_end']:
                    sentence.insert(0, [self.config['START_token'], self.config['START_token']])
                    sentence.append([self.config['END_token'], self.config['END_token']])

                # split the sentences if length is too large
                sentence_chunk = list(chunks(sentence, self.config['max_len']))
                for chunk_i, s in enumerate(sentence_chunk):
                    # pad the sentences
                    s = list(s)
                    for _ in range(max(0, self.config['max_len'] - len(s))):
                        s.append((self.config['PAD_token'], self.config['PAD_token']))
                    tmp_x = []
                    tmp_y = []
                    for x in s:
                        tmp_x.append(self.word2index[x[0]])
                        tmp = np.zeros(self.n_labels)
                        tmp[self.label2index[x[1]]] = 1
                        tmp_y.append(tmp)

                    self.X[k].append(np.array(tmp_x))
                    self.Y[k].append(np.array(tmp_y))
            self.X[k] = np.array(self.X[k])
            self.Y[k] = np.array(self.Y[k])
            print(k)
            print(self.X[k].shape)
            print(self.Y[k].shape)
            n_unknown = (self.X[k][:] == self.word2index[self.config['OOV_token']]).sum()
            print("unknown words {} ({:.2%})".format(n_unknown, n_unknown / (self.X[k].shape[0] * self.X[k].shape[1])))

    def peek(self):
        # no information rate
        # todo
        # ambiguous words?
        self.word2label = defaultdict(list)
        for s in self.X_raw["trn"]:
            for word, label in s:
                if word in self.vocab:
                    self.word2label[word].append(label)
        for word, labels in self.word2label.items():
            self.word2label[word] = Counter(labels).most_common()

        self.ambiguous_words = set([w for w, C in self.word2label.items() if len(C) > 1])

        print("n ambiguous words:", len(self.ambiguous_words))
        print("total words", len(self.word2label))
        print("average meanings if ambiguous", np.array([len(C)
                                                         for w, C in self.word2label.items() if len(C) > 1]).mean())

    def add_absolute_positions(self):
        for i, k in enumerate(["trn", "dev", "tst"]):
            shape = self.X[k].shape
            positions = np.tile(np.array(list(range(shape[1]))), shape[0]).reshape(shape)
            self.X[k] = [self.X[k], positions]

    def load_standardsplit(self):
        print("Loading Universal Dependencies " + self.config['data_path'])
        self.X_raw = defaultdict(list)
        # get list of files
        filenames = {}
        self.part_mapping = {"trn": "train",
                             "dev": "dev",
                             "tst": "test"}
        for file in os.listdir(self.config['data_path']):
            for part, part_conll in self.part_mapping.items():
                mymatch = re.match(r".+?" + part_conll + "\.conllu", file)
                if mymatch is not None:
                    filenames[part] = file

        assert len(filenames) == 3, "Some file is missing..."
        print(filenames)
        for part in self.part_mapping:
            print(part)
            # for part in ["dev"]:
            corpus = pyconll.load_from_file(os.path.join(self.config['data_path'], filenames[part]))
            for sentence in corpus:
                tmp_sentence = []
                for token in sentence:
                    if "-" in token.id:
                        continue
                    if self.config['upos']:
                        tmp_sentence.append([token.form, token.upos])
                    else:
                        tmp_sentence.append([token.form, token.xpos])
                if self.config['shuffle_seqs']:
                    random.shuffle(tmp_sentence)
                self.X_raw[part].append(np.array(tmp_sentence))
        for part in ["trn", "dev", "tst"]:
            # for part in ["dev"]:
            self.X_raw[part] = np.array(self.X_raw[part])
            print(part, self.X_raw[part].shape)

    def compute_sample_weights(self):
        if isinstance(self.X['trn'], list):
            tmp_X = self.X['trn'][0]
        else:
            tmp_X = self.X['trn']
        self.sample_weights = np.ones_like(tmp_X)
        self.sample_weights[tmp_X == self.label2index[self.config['PAD_token']]] = 0.0

    def load_word_embeddings(self):
        print("preparing the embedding matrix")
        W, X = self.load_fasttext(self.config['pretrained_embeddings'], load_first_n=100000)
        assert X.shape[1] == self.config['embed_dim'], "pretrained embeddings have different dimensions " + \
            str(X.shape[1]) + " vs. " + str(self.config['embed_dim'])
        # create the embedding matrix
        # E = np.random.normal(loc=0.0, scale=0.025, size=(self.n_vocab, X.shape[1]))
        E = np.random.uniform(-0.05, 0.05, size=(self.n_vocab, X.shape[1]))
        for word in tqdm(set(list(self.word2index.keys())) & set(W)):
            E[self.word2index[word], :] = X[W.index(word), :]
        self.embedding_matrix = E

    @staticmethod
    def load_fasttext(fname, load_first_n=None):
        print("loading fasttext", fname)

        def load_vectors(fname):
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            n, d = map(int, fin.readline().split())
            data = {}
            count = 0
            for line in tqdm(fin):
                count += 1
                if load_first_n is not None and count > load_first_n:
                    break
                tokens = line.rstrip().split(' ')
                if len(tokens) == 2:
                    continue
                data[tokens[0]] = list(map(float, tokens[1:]))
            return data

        fb = load_vectors(fname)
        W = list(fb.keys())
        W = [x for x in W]
        X = np.vstack(tuple([fb[x] for x in W]))

        return W, X

    def get_vocabulary_char(self):
        vocab = []
        if self.config['build_voc_on_all_char']:
            for_voc = self.X_raw['trn'].tolist() + self.X_raw['dev'].tolist() + self.X_raw['tst'].tolist()
        else:
            for_voc = self.X_raw['trn'].tolist()
        for s in for_voc:
            for word in s:
                vocab.extend([c for c in word[0]])

        vocab.extend(list(self.config['END_token']))
        vocab.extend(list(self.config['START_token']))
        vocab.extend(list(self.config['OOV_token']))
        vocab.extend(list(self.config['PAD_token']))
        vocab = Counter(vocab)
        self.n_vocab_orig_char = len(vocab)
        if self.config['num_voc'] == 0:
            self.config['num_voc'] = self.n_vocab_orig // 2
        if self.config['num_voc_char'] != -1:
            vocab = vocab.most_common(n=self.config['num_voc_char'])
        else:
            vocab = vocab.most_common()
        print("n dictionary char:", len(vocab))

        vocab = [x[0] for x in vocab]

        if self.config['start_end_char']:
            vocab.insert(0, self.config['END_token'])
            vocab.insert(0, self.config['START_token'])

        vocab.insert(0, self.config['OOV_token'])
        vocab.insert(0, self.config['PAD_token'])

        self.index2char = dict(enumerate(vocab))

        self.char2index = defaultdict(lambda: 1, invdict(self.index2char))

        self.vocab_char = set(vocab)

        self.n_vocab_char = len(self.char2index)

    def transform_indices_char(self):
        self.X_char = {}

        self.lengths_char = []

        for i, k in enumerate(["trn", "dev", "tst"]):
            self.X_char[k] = []
            for sentence in self.X_raw[k]:
                sentence = sentence.tolist()

                if self.config['start_end']:
                    sentence.insert(0, [self.config['START_token'], self.config['START_token']])
                    sentence.append([self.config['END_token'], self.config['END_token']])

                # split the sentences if length is too large
                sentence_chunk = list(chunks(sentence, self.config['max_len']))
                for chunk_i, s in enumerate(sentence_chunk):
                    self.X_char[k].append([])
                    # pad the sentences
                    s = list(s)
                    for _ in range(max(0, self.config['max_len'] - len(s))):
                        s.append((self.config['PAD_token'], self.config['PAD_token']))
                    for word in s:
                        word = list(word[0])

                        self.lengths_char.append(len(word))
                        if self.config['start_end_char']:
                            word.insert(0, [self.config['START_token'], self.config['START_token']])
                            word.append([self.config['END_token'], self.config['END_token']])

                        # split the word if length is too large
                        word_chunk = list(chunks(word, self.config['max_len_char']))
                        for chunk_j, w in enumerate(word_chunk):
                            # pad the words
                            w = list(w)
                            for _ in range(max(0, self.config['max_len_char'] - len(w))):
                                w.append((self.config['PAD_token'], self.config['PAD_token']))
                            tmp_x = []
                            for x in w:
                                tmp_x.append(self.char2index[x[0]])

                            self.X_char[k][-1].append(np.array(tmp_x))
                            # only consider the first part of the word
                            # if a word is longer than max_len_char, then the rest of it is ignored
                            break
                    self.X_char[k][-1] = np.array(self.X_char[k][-1])

            self.X_char[k] = np.array(self.X_char[k])
            print("CHARACTER LEVEL")
            print(k)
            print(self.X_char[k].shape)
            n_unknown = (self.X_char[k].flatten() == self.char2index[self.config['OOV_token']]).sum()
            print("unknown characters {} ({:.2%})".format(n_unknown, n_unknown / (len(self.X_char[k].flatten()))))
            if self.config['positions']:
                self.X[k].append(self.X_char[k])
            else:
                self.X[k] = [self.X[k], self.X_char[k]]
