import os 
import sys 
import tensorflow as tf 
from tensorflow.keras import layers, Model

# class Layer()

class MotionEncoder(tf.keras.Model):
    
    def __init__(self, output_dims, hidden_dims, activation=tf.nn.elu, name=None):
        super(MotionEncoder, self).__init__(name)
        self.layer1 = layers.Dense(hidden_dims, activation=activation)
        self.output_layer = layers.Dense(output_dims, activation=None)
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs, training=False):
        if training:
            inputs = self.dropout(inputs)
        layer1_output = self.layer1(inputs)
        if training:
            layer1_output = self.dropout(layer1_output)
        output = self.output_layer(layer1_output)
        return output


class MotionDecoder(tf.keras.Model):

    def __init__(self, output_dims, hidden_dims, activation=tf.nn.elu, name=None):
        super(MotionDecoder, self).__init__(name)
        self.layer1 = layers.Dense(hidden_dims, activation=activation)
        self.output_layer = layers.Dense(output_dims, activation=None)
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs, training=False):
        if training:
            inputs = self.dropout(inputs)
        layer1_output = self.layer1(inputs)
        if training:
            layer1_output = self.dropout(layer1_output)
        output = self.output_layer(layer1_output)
        return output

class MotionEnDecoder(tf.keras.Model):

    def __init__(self,encoder, decoder, name=None):
        super(MotionEnDecoder, self).__init__(name=name)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        encodered_data = self.encoder(inputs)
        return self.decoder(encodered_data)




    

            