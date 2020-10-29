import os

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from keras import optimizers
import tensorflow as tf
import keras.backend as K
import pandas as pd
from .callbacks import BatchCSVLogger


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


from utils.utils import *


class KModel(object):
    def __init__(self, config):
        self.config = config

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config['gpu_id']

        if "memory_fraction" not in self.config:
            self.config["memory_fraction"] = 0.9
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.config['memory_fraction'])
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        # gpu_config = tf.ConfigProto(gpu_options=gpu_options)
        self.session = tf.Session(config=gpu_config)
        K.set_session(self.session)

        if 'log_dir' in self.config:
            for mydir in [self.config['log_dir'], self.config['log_dir'] + "cp/", self.config['log_dir'] + "cs/", self.config['log_dir'] + "csb/"]:
                if not os.path.exists(mydir):
                    mkdir(mydir)

        self.callbacks = []

    def set_callbacks(self, X, Y):
        self.callbacks = []
        cp = ModelCheckpoint(self.config['log_dir'] + "cp/" + self.config['ex_id'] + ".hdf5",
                             monitor=self.config['early_stopping_metric'],
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='max',
                             period=1)
        self.callbacks.append(cp)
        cs = CSVLogger(self.config['log_dir'] + "cs/" + self.config['ex_id'], separator=',', append=False)
        self.callbacks.append(cs)
        csbatch = BatchCSVLogger(self.config['log_dir'] + "csb/" +
                                 self.config['ex_id'], separator=',', append=False, X=X, Y=Y)
        self.callbacks.append(csbatch)
        if 'early_stopping' in self.config and self.config['early_stopping']:
            es = EarlyStopping(monitor=self.config['early_stopping_metric'],
                               min_delta=0,
                               patience=self.config['early_stopping_patience'],
                               verbose=0,
                               mode='auto')
            self.callbacks.append(es)
        # tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        # self.callbacks.append(tb)

    def set_optimizer(self):
        if self.config['optimizer'] == 'adam':
            self.optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    def train_batch(self, X, Y):
        self.model.train_on_batch(X, Y)

    def train(self, X, Y, **kwargs):
        self.model.fit(X, Y,
                       epochs=self.config['n_epochs'],
                       callbacks=self.callbacks,
                       **kwargs)

    def train_gen(self, generator, **kwargs):
        self.model.fit_generator(generator,
                                 epochs=self.config['n_epochs'],
                                 callbacks=self.callbacks,
                                 **kwargs)

    def plot(self):
        if 'log_dir' not in self.config:
            print("No logs specified.")
            return
        pdf = matplotlib.backends.backend_pdf.PdfPages(self.config['log_dir'] + self.config['ex_id'] + ".pdf")
        data = pd.read_csv(self.config['log_dir'] + "cs/" + self.config['ex_id'])
        available = list(data.columns)
        for metric in ["loss"] + self.config['metrics']:
            if not isinstance(metric, str):
                metric = metric.__name__
            plt.plot(data[metric])
            if "val_" + metric in available:
                plt.plot(data["val_" + metric])
            plt.title(metric)
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            pdf.savefig(plt.gcf())
            plt.close()
        pdf.close()

    def evaluate(self, X, Y, save=False, extension=""):
        res = self.model.evaluate(X, Y)
        if save:
            outfile = store(self.config['log_dir'] + self.config['ex_id'] + "__" + extension + ".result")
        for k, v in zip(self.model.metrics_names, res):
            show = "{:<10} {:.2f}".format(k, v)
            print(show)
            if save:
                outfile.write(show + "\n")

    def predict(self, X, Y):
        Y_hat = self.model.predict(X, batch_size=2 ^ 30)
        return X, Y, Y_hat

    def store(self):
        model_json = self.model.to_json()
        with open(self.config['log_dir'] + self.config['ex_id'] + "_model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(self.config['log_dir'] + self.config['ex_id'] + "_weights.h5")
        print("Saved model to disk")

    def load(self, path):
        self.model = load_model(path)

    def protocol(self):
        self.config['GITHASH'] = MYGLOBALHASH
        self.config['CMD'] = MYGLOBALCMD
        outfile = store(self.config['log_dir'] + self.config['ex_id'] + ".config")
        json.dump(self.config, outfile, default=lambda o: '<not serializable>')
