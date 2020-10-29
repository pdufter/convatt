import argparse

from utils.utils import *
from data.ptb import *
from data.univdep import *
from models.seqlabel import *
from evaluation.postagging import *

import random
import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser(description='Investigate Position.')

# infrastucture
parser.add_argument("--gpu_id", type=str, default="0", help="")
parser.add_argument("--memory_fraction", type=float, default=0.9, help="")
parser.add_argument("--explore", type=bool_flag, default=False, help="")

# I/O
parser.add_argument("--log_dir", type=str, default="/mounts/work/philipp/positions/logs/", help="")
parser.add_argument("--logging", type=bool_flag, default=True, help="")
parser.add_argument("--ex_id", type=str, default="", help="")
parser.add_argument("--data_path", type=str, default="/mounts/Users/cisintern/philipp/nltk_data/corpora/ptb/", help="")

# data
parser.add_argument("--OOV_token", type=str, default="<OOV>", help="")
parser.add_argument("--PAD_token", type=str, default="<PAD>", help="")
parser.add_argument("--START_token", type=str, default="<S>", help="")
parser.add_argument("--END_token", type=str, default="</S>", help="")
parser.add_argument("--start_end", type=bool_flag, default=False, help="")
parser.add_argument("--shuffle_seqs", type=bool_flag, default=False, help="")
parser.add_argument("--num_voc", type=int, default=0, help="-1: take all, 0: take half")
parser.add_argument("--build_voc_on_all", type=bool_flag, default=True, help="")
parser.add_argument("--max_len", type=int, default=60, help="")
parser.add_argument("--data_split", type=str, default="""[["trn", 0.7],["dev", 0.15],["tst", 0.15]]""", help="")
parser.add_argument("--data_split_path", type=str, default="/mounts/Users/cisintern/philipp/nltk_data/corpora/ptb/data_split.txt", help="")
parser.add_argument("--use_standard_split", type=bool_flag, default=True, help="")
parser.add_argument("--pretrained_embeddings", type=str, default="", help="")
parser.add_argument("--finetune_embeddings", type=bool_flag, default=True, help="")
parser.add_argument("--mergedict_path", type=bool_flag, default=True, help="")
parser.add_argument("--is_univ_dep", type=bool_flag, default=False, help="")
parser.add_argument("--upos", type=bool_flag, default=True, help="")
# data character level
parser.add_argument("--num_voc_char", type=int, default=-1, help="-1: all")
parser.add_argument("--build_voc_on_all_char", type=bool_flag, default=False, help="")
parser.add_argument("--max_len_char", type=int, default=20, help="")
parser.add_argument("--start_end_char", type=bool_flag, default=False, help="")
parser.add_argument("--add_char_level", type=bool_flag, default=True, help="")
parser.add_argument("--finetune_embeddings_char", type=bool_flag, default=True, help="")


# model
parser.add_argument("--model", type=str, default="cnn", help="")
parser.add_argument("--comment", type=str, default="debug", help="")
parser.add_argument("--optimizer", type=str, default="rmsprop", help="")
parser.add_argument("--n_epochs", type=int, default=100, help="")
parser.add_argument("--batch_size", type=int, default=32, help="")
parser.add_argument("--positions", type=str, default="", help="")
parser.add_argument("--positions_mode", type=str, default="concatenate", help="add or concatenate")
parser.add_argument("--embed_dim", type=int, default="128", help="")
parser.add_argument("--early_stopping", type=bool_flag, default=True, help="")
parser.add_argument("--early_stopping_patience", type=int, default=3, help="")
parser.add_argument("--early_stopping_metric", type=str, default="val_true_acc", help="")
parser.add_argument("--sample_weights", type=bool_flag, default=True, help="")
parser.add_argument("--selfatt_residuals", type=str, default="add", help="")
parser.add_argument("--crf", type=bool_flag, default=False, help="")
parser.add_argument("--n_layers", type=int, default=4, help="")
parser.add_argument("--n_hidden_units", type=int, default=-1, help="")
parser.add_argument("--cnn_filter_width", type=int, default=3, help="")
parser.add_argument("--activation_function", type=str, default="relu", help="")
parser.add_argument("--l2_regularisation", type=float, default=0.00, help="")
parser.add_argument("--dropout_rate", type=float, default=0.1, help="")
parser.add_argument("--optim_args", type=str, default="lr=f0.001", help="")
parser.add_argument("--n_attention_heads", type=int, default=4, help="")
parser.add_argument("--position_embed_dim", type=int, default=64, help="")
parser.add_argument("--embedding_mask_zero", type=bool_flag, default=False, help="")
parser.add_argument("--input_masking", type=bool_flag, default=False, help="")
parser.add_argument("--finetune_position_embeddings", type=bool_flag, default=True, help="")
parser.add_argument("--residual_for_last_layer", type=bool_flag, default=True, help="")
parser.add_argument("--abs_positions_within_attention", type=bool_flag, default=False, help="")
parser.add_argument("--rel_positions_within_attention", type=bool_flag, default=False, help="")
parser.add_argument("--fancy_positions_within_attention", type=bool_flag, default=False, help="")
# model char
parser.add_argument("--model_char", type=str, default="cnn", help="")
parser.add_argument("--positions_char", type=str, default="", help="")
parser.add_argument("--embed_dim_char", type=int, default="64", help="")
parser.add_argument("--weight_normalization", type=bool_flag, default=False, help="")
parser.add_argument("--E_normalization", type=bool_flag, default=False, help="")

parser.add_argument("--seed", type=int, default="42", help="")



###################
# parse parameters and check values
config_class = parser.parse_args()
config = config_class.__dict__

random.seed(config["seed"])
np.random.seed(config["seed"])
tf.set_random_seed(config["seed"])

if config['logging']:
    if config['ex_id'] == "":
        ex_id = get_randomid()
        while os.path.exists(config['log_dir'] + ex_id):
            ex_id = get_randomid()
        config['ex_id'] = ex_id
    else:
        if os.path.exists(config['log_dir'] + config['ex_id']):
            raise ValueError(config['log_dir'] + config['ex_id'] + " exists already.")
    open(config['log_dir'] + config['ex_id'], 'w').close()
else:
    os.system('rm ' + config['log_dir'] + "tmp*")
    os.system('rm ' + config['log_dir'] + "cs/tmp*")
    os.system('rm ' + config['log_dir'] + "cp/tmp*")
    config['ex_id'] = "tmp"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']

config['true_acc'] = get_early_stopping_metric(config)
config['metrics'] = ['acc', config['true_acc']]
if config['sample_weights']:
    config['sample_weight_mode'] = 'temporal'
else:
    config['sample_weight_mode'] = None

if config['sample_weights']:
    assert config['crf'] is False, "sample weigths not compatible with crf"

if config['optim_args']:
    optim_args = {}
    for elem in config['optim_args'].split(","):
        k, v = elem.split("=")
        if v[0] == 'b':
            v = bool(v[1:])
        elif v[0] == 'f':
            v = float(v[1:])
        elif v[0] == 'i':
            v = int(v[1:])
        elif v[0] == 's':
            v = str(v[1:])
        optim_args[k] = v
    config['optim_args'] = optim_args
else:
    config['optim_args'] = {}

if config['n_hidden_units'] == -1:
    config['n_hidden_units'] = config['embed_dim'] + int(bool(config['positions'])) * config['position_embed_dim'] + int(bool(config['add_char_level'])) * config['embed_dim_char']
    print("DIMENSIONS:")
    print("embeddings: ", config['embed_dim'])
    print("positions: ", int(bool(config['positions'])) * config['position_embed_dim'])
    print("characters: ", int(bool(config['add_char_level'])) * config['embed_dim_char'])
    print("hidden: ", config['n_hidden_units'])

assert config['position_embed_dim'] % 2 == 0, "pos_dim not even number"

assert config['n_hidden_units'] % config['n_attention_heads'] == 0, "hidden units and attention heads not compatible"

###################
# Data
if config['is_univ_dep']:
    data = UD(config)
    data.load_standardsplit()
else:
    data = PT(config)
    if config['use_standard_split']:
        data.load_WSJ_standardsplit()
    else:
        data.load_data()
        data.create_data_split()
        data.split_data_loaded()
data.get_vocabulary()
data.transform_indices()
data.peek()
if config['positions']:
    data.add_absolute_positions()
if config['sample_weights']:
    data.compute_sample_weights()
if config['pretrained_embeddings']:
    data.load_word_embeddings()
if config['add_char_level']:
    data.get_vocabulary_char()
    data.transform_indices_char()
    # not implemented or used yet
    #data.add_absolute_positions_char()

# store some parameters in config for easier analysis
config['n_vocab_true'] = data.n_vocab
config['n_labels_true'] = data.n_labels
if config['add_char_level']:
    config['n_vocab_char_true'] = data.n_vocab_char

###################
# Models



if config['model'] == 'majority_baseline':
    prediction = {}
    for k, v in data.word2label.items():
        prediction[k] = v[0][0]
    prediction[config['OOV_token']] = config['OOV_token']
    prediction[config['PAD_token']] = config['PAD_token']
    # now predict on dev and tst
    X_dev = data.X["dev"].flatten()
    Y_hat_dev = []
    for elem in X_dev:
        pred = data.label2index[prediction[data.index2word[elem]]]
        tmp = np.zeros(data.n_labels)
        tmp[pred] = 1
        Y_hat_dev.append(tmp)
    Y_hat_dev = np.array(Y_hat_dev)
    Y_hat_dev = np.reshape(Y_hat_dev, data.Y['dev'].shape)

    X_tst = data.X["tst"].flatten()
    Y_hat_tst = []
    for elem in X_tst:
        pred = data.label2index[prediction[data.index2word[elem]]]
        tmp = np.zeros(data.n_labels)
        tmp[pred] = 1
        Y_hat_tst.append(tmp)
    Y_hat_tst = np.array(Y_hat_tst)
    Y_hat_tst = np.reshape(Y_hat_tst, data.Y['tst'].shape)

else:
    mod = SeqLabelModel(config, data)
    mod.set_optimizer()
    mod.get_model()
    mod.set_callbacks(data.X["dev"], data.Y["dev"])
    mod.train(data.X["trn"], data.Y["trn"], validation_data=(data.X["dev"], data.Y["dev"]), batch_size=config['batch_size'], sample_weight=data.sample_weights, verbose=1)
    mod.model.load_weights(mod.config['log_dir'] + "cp/" + config['ex_id'] + ".hdf5")
    mod.plot()
    mod.store()
    mod.evaluate(data.X["dev"], data.Y["dev"], save=True, extension="dev")
    mod.evaluate(data.X["tst"], data.Y["tst"], save=True, extension="tst")
    mod.protocol()
    _, _, Y_hat_dev = mod.predict(data.X["dev"], data.Y["dev"])
    _, _, Y_hat_tst = mod.predict(data.X["tst"], data.Y["tst"])

if isinstance(data.X["dev"], list):
    X_dev = data.X["dev"][0]
    X_tst = data.X["tst"][0]
else:
    X_dev = data.X["dev"]
    X_tst = data.X["tst"]

############
# Evaluation

eval_dev = Evaluation(X_dev, data.Y["dev"], Y_hat_dev, data, config)
#eval_dev = Evaluation(X_dev[:100, :], data.Y["dev"][:100, :], Y_hat_dev, data)
res_dev = eval_dev.get_summary(config['log_dir'] + config['ex_id'] + "_dev.txt")

eval_tst = Evaluation(X_tst, data.Y["tst"], Y_hat_tst, data, config)
res_tst = eval_tst.get_summary(config['log_dir'] + config['ex_id'] + "_tst.txt")


if config['logging']:
    global_log = open(config['log_dir'] + "overview.txt", 'a')
    digits = 4
    global_log.write(config['ex_id'] + "\t" + config['comment'] + "\t")
    global_log.write(str(round(res_dev['all,noPAD'], digits)) + "\t")
    global_log.write(str(round(res_dev['OOV'], digits)) + "\t")
    global_log.write(str(round(res_dev['PAD'], digits)) + "\t")
    global_log.write(str(round(res_dev['ambig,noPAD,noOOV'], digits)) + "\t")
    global_log.write(str(round(res_dev['unambig,noPAD,noOOV'], digits)) + "\n")
    global_log.close()

