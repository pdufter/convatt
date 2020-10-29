import numpy as np
import keras.backend as K
import tensorflow as tf

from utils.utils import *


def get_early_stopping_metric(config):
    if config['start_end']:
        def true_acc(y_true, y_pred):
            y_true = K.flatten(K.argmax(y_true[:, 1:-1], axis=-1))
            y_pred = K.flatten(K.argmax(y_pred[:, 1:-1], axis=-1))
            mask = K.not_equal(y_true, K.zeros_like(y_true))
            correct = K.cast(K.equal(tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred, mask)), 'float32')
            return K.sum(correct[:]) / K.cast(K.shape(correct)[0], 'float32')
    else:
        def true_acc(y_true, y_pred):
            y_true = K.flatten(K.argmax(y_true, axis=-1))
            y_pred = K.flatten(K.argmax(y_pred, axis=-1))
            mask = K.not_equal(y_true, K.zeros_like(y_true))
            correct = K.cast(K.equal(tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred, mask)), 'float32')
            return K.sum(correct[:]) / K.cast(K.shape(correct)[0], 'float32')
    return true_acc


class Evaluation(object):
    """docstring for Evaluation"""

    def __init__(self, X, Y, Y_hat, data, config):
        super(Evaluation, self).__init__()

        # if start end, then get rid of that!
        if config['start_end']:
            self.X = X[:, 1:-1]
            self.Y = Y[:, 1:-1]
            self.Y_hat = Y_hat[:, 1:-1]
        else:
            self.X = X
            self.Y = Y
            self.Y_hat = Y_hat
        self.data = data

    def get_summary(self, filename):
        result = {}
        X = self.X.flatten()
        Y = self.Y.argmax(axis=-1).flatten()
        if len(self.Y_hat.shape) < 3:
            Y_hat = self.Y_hat.flatten()
        else:
            Y_hat = self.Y_hat.argmax(axis=-1).flatten()

        correct = Y == Y_hat
        result['all'] = sum(correct) / max(len(correct), 1)

        mask = X != 0
        correct = Y[mask] == Y_hat[mask]
        result['all,noPAD'] = sum(correct) / max(len(correct), 1)

        mask = ~np.logical_or(X == 0, X == 1)
        correct = Y[mask] == Y_hat[mask]
        result['all,noPADnoOOV'] = sum(correct) / max(len(correct), 1)

        mask = X == 1
        correct = Y[mask] == Y_hat[mask]
        result['OOV'] = sum(correct) / max(len(correct), 1)

        mask = X == 0
        correct = Y[mask] == Y_hat[mask]
        result['PAD'] = sum(correct) / max(len(correct), 1)

        ambiguous_indices = set([self.data.word2index[x] for x in self.data.ambiguous_words])
        mask = np.array([x in ambiguous_indices and x != 0 and x != 1 for x in X])
        correct = Y[mask] == Y_hat[mask]
        result['ambig,noPAD,noOOV'] = sum(correct) / max(len(correct), 1)

        mask = ~np.array([x in ambiguous_indices or x == 0 or x == 1 for x in X])
        correct = Y[mask] == Y_hat[mask]
        result['unambig,noPAD,noOOV'] = sum(correct) / max(len(correct), 1)

        for tag, taglabel in self.data.index2label.items():
            mask = np.array([y == tag for y in Y])
            correct = Y[mask] == Y_hat[mask]
            result[taglabel] = sum(correct) / max(len(correct), 1)

        outfile = store(filename)
        to_write = ""
        for k, v in result.items():
            to_write += str(k) + " " + str(v) + "\n"
        outfile.write(to_write)
        outfile.close()
        print(to_write)
        return result
