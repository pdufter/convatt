from keras import backend as K
from keras.layers import *
from keras import regularizers
import tensorflow as tf

from keras.engine.topology import Layer
from keras.activations import softmax
from keras.models import Sequential

import keras

from models.kmodel import *
from models.selfattention import *
from keras_layer_normalization import LayerNormalization


class SeqLabelModel(KModel):
    def __init__(self, config, data):
        super(SeqLabelModel, self).__init__(config)
        self.data = data
        self.config = config

    def get_model(self):
        inputs = Input(shape=(self.config['max_len'], ))

        if self.config['pretrained_embeddings']:
            embed_layer = Embedding(output_dim=self.config['embed_dim'],
                                    input_dim=self.config['n_vocab_true'],
                                    input_length=self.config['max_len'], weights=[
                                    self.data.embedding_matrix], mask_zero=self.config['embedding_mask_zero'])
        else:
            embed_layer = Embedding(output_dim=self.config['embed_dim'], input_dim=self.config['n_vocab_true'],
                                    input_length=self.config['max_len'], mask_zero=self.config['embedding_mask_zero'])

        embed_layer.trainable = self.config['finetune_embeddings']
        embed = embed_layer(inputs)

        if self.config['positions'] == 'embeddings':
            positions = Input(shape=(self.config['max_len'], ))
            positions_embed = Embedding(output_dim=self.config['position_embed_dim'],
                                        input_dim=self.config['max_len'],
                                        input_length=self.config['max_len'],
                                        mask_zero=self.config['embedding_mask_zero'],
                                        trainable=self.config['finetune_position_embeddings'])(positions)
            if self.config['positions_mode'] == 'concatenate':
                embed = Concatenate()([embed, positions_embed])
            elif self.config['positions_mode'] == 'add':
                embed = Add()([embed, positions_embed])
            else:
                raise ValueError("Unknonw position_mode")
        # add char level processing
        if self.config['add_char_level']:
            inputs_char = Input(shape=(self.config['max_len'], self.config['max_len_char'], ))

            embed_layer_char = TimeDistributed(Embedding(
                output_dim=self.config['embed_dim_char'], input_dim=self.config['n_vocab_char_true'],
                input_length=self.config['max_len_char'], mask_zero=self.config['embedding_mask_zero']))

            embed_layer_char.trainable = self.config['finetune_embeddings_char']
            embed_char = embed_layer_char(inputs_char)

            # do some CNN
            cnn1_char = TimeDistributed(Conv1D(self.config['embed_dim_char'], 5,
                                               activation='relu', padding='same', kernel_regularizer=regularizers.l2(
                self.config['l2_regularisation'])))(embed_char)
            cnn2_char = TimeDistributed(Conv1D(self.config['embed_dim_char'], 5,
                                               activation='relu', padding='same', kernel_regularizer=regularizers.l2(
                self.config['l2_regularisation'])))(cnn1_char)
            cnn3_char = TimeDistributed(Conv1D(self.config['embed_dim_char'], 5,
                                               activation='relu', padding='same', kernel_regularizer=regularizers.l2(
                self.config['l2_regularisation'])))(cnn2_char)
            cnn3_char = Lambda(lambda x: K.permute_dimensions(x, (0, 1, 3, 2)))(cnn3_char)
            hidden_char = TimeDistributed(GlobalMaxPooling1D(data_format='channels_first'))(cnn3_char)
            # then maxpooling, then concatenate
            embed = Concatenate()([embed, hidden_char])

        hidden = embed

        n_heads = self.config['n_attention_heads']
        att_dim = int(self.config['n_hidden_units'] / n_heads)

        def MultiHeadSelfAttention(tmp_input, att_dim, n_heads, add_abs_position=True, add_rel_position=True):
            att = []
            for _ in range(n_heads):
                A_1 = SelfAttention(att_dim,
                                    add_abs_position=add_abs_position and self.config['abs_positions_within_attention'],
                                    add_rel_position=add_rel_position and self.config['rel_positions_within_attention'],
                                    weight_normalization=self.config['weight_normalization'])(tmp_input)
                att.append(A_1)
            multiheadatt = Concatenate()(att)
            output = Add()([tmp_input, multiheadatt])
            return output

        def MultiHeadSelfAttentionconv(tmp_input, att_dim, n_heads, add_abs_position=True, add_rel_position=True):
            att = []
            for _ in range(n_heads):
                A_1_tmp = SelfAttentionPart1(att_dim,
                                             add_abs_position=self.config['abs_positions_within_attention'],
                                             add_rel_position=self.config['rel_positions_within_attention'],
                                             weight_normalization=self.config['weight_normalization'])(tmp_input)
                A_1_conv = Conv1D(self.config['max_len'], self.config['cnn_filter_width'],
                                  activation=self.config['activation_function'], padding='same')(A_1_tmp)
                A_1 = SelfAttentionPart2(att_dim,
                                         weight_normalization=self.config['weight_normalization'])(
                    [tmp_input, A_1_conv])
                att.append(A_1)
            multiheadatt = Concatenate()(att)
            output = Add()([tmp_input, multiheadatt])
            return output

        def MultiHeadSelfAttentionconv2d(tmp_input, att_dim, n_heads, add_abs_position=True, add_rel_position=True):
            att = []
            for _ in range(n_heads):
                A_1_tmp = SelfAttentionPart1(att_dim,
                                             add_abs_position=self.config['abs_positions_within_attention'],
                                             add_rel_position=self.config['rel_positions_within_attention'],
                                             weight_normalization=self.config['weight_normalization'])(tmp_input)
                A_1_tmp = Lambda(lambda x: keras.backend.expand_dims(x, axis=1))(A_1_tmp)
                A_1_conv = Conv2D(1, self.config['cnn_filter_width'],
                                  activation=self.config['activation_function'], padding='same', data_format="channels_first")(A_1_tmp)
                A_1_conv = Lambda(lambda x: keras.backend.squeeze(x, axis=1))(A_1_conv)
                A_1 = SelfAttentionPart2(att_dim,
                                         weight_normalization=self.config['weight_normalization'])(
                    [tmp_input, A_1_conv])
                att.append(A_1)
            multiheadatt = Concatenate()(att)
            output = Add()([tmp_input, multiheadatt])
            return output

        for i in range(self.config['n_layers']):
            if i == 0:
                add_abs_position = True
                add_rel_position = True
            else:
                add_abs_position = False
                add_rel_position = False
            hidden = Dropout(self.config['dropout_rate'])(hidden)
            if self.config['model'] == 'selfattention':
                hidden = MultiHeadSelfAttention(hidden, att_dim, n_heads,
                                                add_abs_position=add_abs_position, add_rel_position=add_rel_position)
            elif self.config['model'] == 'selfattention_experimental':
                hidden = MultiHeadSelfAttentionconv(
                    hidden, att_dim, n_heads, add_abs_position=add_abs_position, add_rel_position=add_rel_position)
            elif self.config['model'] == 'selfattention_experimental2d':
                hidden = MultiHeadSelfAttentionconv2d(
                    hidden, att_dim, n_heads, add_abs_position=add_abs_position, add_rel_position=add_rel_position)

        if self.config['residual_for_last_layer']:
            hidden = Concatenate()([embed, hidden])
        outputs = TimeDistributed(Dense(self.config['n_labels_true'], activation='softmax'))(hidden)
        loss = 'categorical_crossentropy'

        if self.config['positions'] and not self.config['add_char_level']:
            self.model = Model(inputs=[inputs, positions], outputs=outputs)
        elif self.config['positions'] and self.config['add_char_level']:
            self.model = Model(inputs=[inputs, positions, inputs_char], outputs=outputs)
        elif not self.config['positions'] and self.config['add_char_level']:
            self.model = Model(inputs=[inputs, inputs_char], outputs=outputs)
        else:
            self.model = Model(inputs=inputs, outputs=outputs)
        if self.config['optimizer'] == 'adam':
            optimizer = keras.optimizers.Adam(**self.config['optim_args'])
        elif self.config['optimizer'] == 'sgd':
            optimizer = keras.optimizers.SGD(**self.config['optim_args'])
        elif self.config['optimizer'] == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(**self.config['optim_args'])
        elif self.config['optimizer'] == 'adagrad':
            optimizer = keras.optimizers.Adagrad(**self.config['optim_args'])
        elif self.config['optimizer'] == 'adadelta':
            optimizer = keras.optimizers.Adadelta(**self.config['optim_args'])
        elif self.config['optimizer'] == 'adamax':
            optimizer = keras.optimizers.Adamax(**self.config['optim_args'])
        elif self.config['optimizer'] == 'nadam':
            optimizer = keras.optimizers.Nadam(**self.config['optim_args'])

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=self.config['metrics'],
                           sample_weight_mode=self.config['sample_weight_mode'])
        print(self.model.summary())
