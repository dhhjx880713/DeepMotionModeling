import tensorflow as tf 
from tensorflow.keras import layers, Model
import numpy as np 
tf.keras.backend.set_floatx('float64')



class SingleLayerPoseEncoder(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



class FullBodyPoseEncoder(tf.keras.Model):

    def __init__(self, output_dim, dropout_rate=0.3, npc=10, activation=tf.nn.elu):
        super(FullBodyPoseEncoder, self).__init__()
        self.L1_units = 512
        self.L2_units = 256
        self.L3_units = 128
        self.L4_units = npc
        self.activation = activation
        self.output_dim = output_dim
        self.L1 = layers.Dense(self.L1_units, activation=self.activation, name='layer1')
        self.L1_inverse = layers.Dense(self.L1_units, activation=self.activation, name='layer1_inverse')
        self.L2 = layers.Dense(self.L2_units, activation=self.activation, name='layer2')
        self.L2_inverse = layers.Dense(self.L2_units, activation=self.activation, name='layer2_inverse')
        self.L3 = layers.Dense(self.L3_units, activation=self.activation, name='layer3')
        self.L3_inverse = layers.Dense(self.L3_units, activation=self.activation, name='layer3_inverse')
        self.L4 = layers.Dense(self.L4_units, activation=self.activation, name='layer4')
        self.L4_inverse = layers.Dense(self.L4_units, activation=self.activation, name='layer4_inverse')
        self.output_layer = layers.Dense(self.output_dim, name='output_layer')
        self.dropout = layers.Dropout(rate=dropout_rate)
    
    def encode_dropout_first(self, inputs, training=None):
        if training:
            inputs = self.dropout(inputs)
        L1_res = self.L1(inputs)
        if training:
            L1_res = self.dropout(L1_res)
        L2_res = self.L2(L1_res)
        if training:
            L2_res = self.dropout(L2_res)
        L3_res = self.L3(L2_res)
        if training:
            L3_res = self.dropout(L3_res)
        L4_res = self.L4(L3_res)
        return L4_res

    def encode(self, inputs, training=None):
        L1_res = self.L1(inputs)
        if training:
            L1_res = self.dropout(L1_res)
        L2_res = self.L2(L1_res)
        if training:
            L2_res = self.dropout(L2_res)
        L3_res = self.L3(L2_res)
        if training:
            L3_res = self.dropout(L3_res)
        L4_res = self.L4(L3_res)
        # if training:
        #     L4_res = self.dropout(L4_res)   
        return L4_res

    def decode_dropout_first(self, inputs, training=None):
        if training:
            inputs = self.dropout(inputs)
        L4_inverse_res = self.L4_inverse(inputs)
        if training:
            L4_inverse_res = self.dropout(L4_inverse_res)
        L3_inverse_res = self.L3_inverse(L4_inverse_res)
        if training:
            L3_inverse_res = self.dropout(L3_inverse_res)
        L2_inverse_res = self.L2_inverse(L3_inverse_res)
        if training:
            L2_inverse_res = self.dropout(L2_inverse_res)
        L1_inverse_res = self.L1_inverse(L2_inverse_res)
        output = self.output_layer(L1_inverse_res)
        return output    

    def decode(self, inputs, training=None):
        L4_inverse_res = self.L4_inverse(inputs)
        if training:
            L4_inverse_res = self.dropout(L4_inverse_res)
        L3_inverse_res = self.L3_inverse(L4_inverse_res)
        if training:
            L3_inverse_res = self.dropout(L3_inverse_res)
        L2_inverse_res = self.L2_inverse(L3_inverse_res)
        if training:
            L2_inverse_res = self.dropout(L2_inverse_res)
        L1_inverse_res = self.L1_inverse(L2_inverse_res)
        if training:
            L1_inverse_res = self.dropout(L1_inverse_res)
        output = self.output_layer(L1_inverse_res)
        return output

    def call(self, inputs, training=None):
        z_encoder = self.encode(inputs, training=training)
        decoder_value = self.decode(z_encoder, training=training)
        return decoder_value
    
    def __call__(self, inputs, training=None):
        print("using dropout first!")
        z_encoder = self.encode_dropout_first(inputs, training=training)
        decoder_value = self.decode_dropout_first(z_encoder, training=training)
        return decoder_value
       