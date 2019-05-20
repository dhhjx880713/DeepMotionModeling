import sys
import numpy as np

sys.path.append(r'../nn')
sys.path.append(r'../')
sys.path.append(r'../motion')

rng = np.random.RandomState(23456)
import os
from Layer import Layer
from HiddenLayer import HiddenLayer
from BiasLayer import BiasLayer
from DropoutLayer import DropoutLayer
from ActivationLayer import ActivationLayer


class ResPhaseFunctionedNetwork(Layer):

    def __init__(self, rng=rng, input_shape=1, output_shape=1, dropout=0.7):
        self.n = 30
        self.nslices = 4
        self.dropout0 = DropoutLayer(dropout, rng=rng)
        self.dropout1 = DropoutLayer(dropout, rng=rng)
        self.dropout2 = DropoutLayer(dropout, rng=rng)
        self.activation = ActivationLayer('ELU')

        self.W0 = HiddenLayer((self.nslices, 512, input_shape-1), rng=rng, gamma=0.01)
        self.W1 = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.W_res = HiddenLayer((self.nslices, 512, 512), rng=rng, gamma=0.01)
        self.W2 = HiddenLayer((self.nslices, output_shape, 512), rng=rng, gamma=0.01)


        self.b0 = BiasLayer((self.nslices, 512))
        self.b1 = BiasLayer((self.nslices, 512))
        self.b_res = BiasLayer((self.nslices, 512))
        self.b2 = BiasLayer((self.nslices, output_shape))

        self.layers = [
            self.W0, self.W1, self.W2,
            self.b0, self.b1, self.b2]
        self.style_layers = [self.W_res, self.b_res]
        self.fix_params = sum([layer.params for layer in self.layers], [])
        self.params = sum([layer.params for layer in self.style_layers], [])

    def __call__(self, input):

        pscale = self.nslices * input[:,-1]
        pamount = pscale % 1.0

        pindex_1 = T.cast(pscale, 'int32') % self.nslices
        pindex_0 = (pindex_1-1) % self.nslices
        pindex_2 = (pindex_1+1) % self.nslices
        pindex_3 = (pindex_1+2) % self.nslices

        Wamount = pamount.dimshuffle(0, 'x', 'x')
        bamount = pamount.dimshuffle(0, 'x')

        W0 = ResPhaseFunctionedNetwork.cubic(self.W0.W[pindex_0], self.W0.W[pindex_1], self.W0.W[pindex_2], self.W0.W[pindex_3], Wamount)
        W1 = ResPhaseFunctionedNetwork.cubic(self.W1.W[pindex_0], self.W1.W[pindex_1], self.W1.W[pindex_2], self.W1.W[pindex_3], Wamount)
        W_res = ResPhaseFunctionedNetwork.cubic(self.W_res.W[pindex_0], self.W_res.W[pindex_1], self.W_res.W[pindex_2], self.W_res.W[pindex_3],
                      Wamount)
        W2 = ResPhaseFunctionedNetwork.cubic(self.W2.W[pindex_0], self.W2.W[pindex_1], self.W2.W[pindex_2], self.W2.W[pindex_3], Wamount)

        b0 = ResPhaseFunctionedNetwork.cubic(self.b0.b[pindex_0], self.b0.b[pindex_1], self.b0.b[pindex_2], self.b0.b[pindex_3], bamount)
        b1 = ResPhaseFunctionedNetwork.cubic(self.b1.b[pindex_0], self.b1.b[pindex_1], self.b1.b[pindex_2], self.b1.b[pindex_3], bamount)
        b_res = ResPhaseFunctionedNetwork.cubic(self.b_res.b[pindex_0], self.b_res.b[pindex_1], self.b_res.b[pindex_2], self.b_res.b[pindex_3],
                      bamount)
        b2 = ResPhaseFunctionedNetwork.cubic(self.b2.b[pindex_0], self.b2.b[pindex_1], self.b2.b[pindex_2], self.b2.b[pindex_3], bamount)

        H0 = input[:,:-1]
        H1 = self.activation(T.batched_dot(W0, self.dropout0(H0)) + b0)
        H2 = self.activation(T.batched_dot(W1, self.dropout1(H1)) + b1 + T.batched_dot(W_res, self.dropout1(H1)) + b_res)
        H3 =                 T.batched_dot(W2, self.dropout2(H2)) + b2

        return H3

    @staticmethod
    def cubic(y0, y1, y2, y3, mu):
        return (
                (-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu * mu * mu +
                (y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * mu * mu +
                (-0.5 * y0 + 0.5 * y2) * mu +
                (y1))

    def cost(self, input):
        input = input[:,:-1]
        costs = 0
        for layer in self.layers+self.style_layers:
            costs += layer.cost(input)
            input = layer(input)
        return costs / len(self.layers)

    def save(self, database, prefix=''):
        for li, layer in enumerate(self.style_layers):
            layer.save(database, '%sL%03i_' % (prefix, li))

    def load_fix_parameters(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.load(database, '%sL%03i_' % (prefix, li))

    def load_style_parameters(self, database, prefix=''):
        for li, layer in enumerate(self.style_layers):
            layer.load(database, '%sL%03i_' % (prefix, li))

    def discretize_network(self, save_path):
        W0n = self.W0.W.get_value()
        W1n = self.W1.W.get_value()
        W2n = self.W2.W.get_value()
        W_res_n = self.W_res.W.get_value()
        b0n = self.b0.b.get_value()
        b1n = self.b1.b.get_value()
        b2n = self.b2.b.get_value()
        b_res_n = self.b_res.b.get_value()

        for i in range(50):
            pscale = self.nslices * (float(i) / 50)
            pamount = pscale % 1.0

            pindex_1 = int(pscale) % self.nslices
            pindex_0 = (pindex_1 - 1) % self.nslices
            pindex_2 = (pindex_1 + 1) % self.nslices
            pindex_3 = (pindex_1 + 2) % self.nslices

            W0 = ResPhaseFunctionedNetwork.cubic(W0n[pindex_0], W0n[pindex_1], W0n[pindex_2], W0n[pindex_3], pamount)
            W1 = ResPhaseFunctionedNetwork.cubic(W1n[pindex_0], W1n[pindex_1], W1n[pindex_2], W1n[pindex_3], pamount)
            W2 = ResPhaseFunctionedNetwork.cubic(W2n[pindex_0], W2n[pindex_1], W2n[pindex_2], W2n[pindex_3], pamount)
            W_res = ResPhaseFunctionedNetwork.cubic(W_res_n[pindex_0], W_res_n[pindex_1], W_res_n[pindex_2], W_res_n[pindex_3], pamount)
            b0 = ResPhaseFunctionedNetwork.cubic(b0n[pindex_0], b0n[pindex_1], b0n[pindex_2], b0n[pindex_3], pamount)
            b1 = ResPhaseFunctionedNetwork.cubic(b1n[pindex_0], b1n[pindex_1], b1n[pindex_2], b1n[pindex_3], pamount)
            b2 = ResPhaseFunctionedNetwork.cubic(b2n[pindex_0], b2n[pindex_1], b2n[pindex_2], b2n[pindex_3], pamount)
            b_res = ResPhaseFunctionedNetwork.cubic(b_res_n[pindex_0], b_res_n[pindex_1], b_res_n[pindex_2], b_res_n[pindex_3], pamount)
            W0.astype(np.float32).tofile(os.path.join(save_path, 'W0_%03i.bin' % i))
            W1.astype(np.float32).tofile(os.path.join(save_path, 'W1_%03i.bin' % i))
            W2.astype(np.float32).tofile(os.path.join(save_path, 'W2_%03i.bin' % i))
            W_res.astype(np.float32).tofile(os.path.join(save_path, 'W_res_%03i.bin' % i))
            b0.astype(np.float32).tofile(os.path.join(save_path, 'b0_%03i.bin' % i))
            b1.astype(np.float32).tofile(os.path.join(save_path, 'b1_%03i.bin' % i))
            b2.astype(np.float32).tofile(os.path.join(save_path, 'b2_%03i.bin' % i))
            b_res.astype(np.float32).tofile(os.path.join(save_path, 'b_res_%03i.bin' % i))