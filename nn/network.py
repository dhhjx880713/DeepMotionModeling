import tensorflow as tf
import abc
import numpy as np
import sys
from datetime import datetime


class Network(object):

    def __init__(self, name, sess=None):
        self.name = name
        self.params = []
        if sess is not None:
            self.sess = sess
        else:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=config) 
        self.saver = None
    
    def get_params(self):
        return self.params

    def save(self, model_file):
        if self.saver is None:
            raise ValueError("No parameters to save. Add parameters to be saved in build")
        else:
            save_path = self.saver.save(self.sess, model_file)
            print("Model saved in file: %s " % save_path)

    def load(self, model_file):
        if self.saver is None:
            raise ValueError("No parameters to save. Add parameters to be saved in build")
        else:
            self.saver.restore(self.sess, model_file)
    
    @abc.abstractmethod
    def build(self):
        return

    def train(self, input_data, output_data, epochs, learning_rate, batchsize, rng=np.random.RandomState(123456)):
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = self.optimizer.minimize(self.cost)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        n_samples = len(input_data)
        last_mean = 0
        for epoch in range(epochs):
            batchinds = np.arange(n_samples // batchsize)
            rng.shuffle(batchinds)
            loss = []
            for idx, batch_idx in enumerate(batchinds):
                self.sess.run(train_op, 
                feed_dict={self.input: input_data[batch_idx * batchsize : (batch_idx + 1) * batchsize],
                           self.output: output_data[batch_idx * batchsize : (batch_idx + 1) * batchsize]})
                loss.append(self.sess.run(self.cost, feed_dict={self.input: input_data[batch_idx * batchsize : (batch_idx + 1) * batchsize],
                            self.output: output_data[batch_idx * batchsize : (batch_idx + 1) * batchsize]}))
                if np.isnan(loss[-1]): return
                if idx % (int(len(batchinds) / 1000) + 1) == 0:
                    sys.stdout.write('\r[Epoch %i]  %0.1f%% mean %.5f' % (epoch, 100 * float(idx) / len(batchinds),
                                                                          np.mean(loss)))
                    sys.stdout.flush()  
            curr_mean = np.mean(loss)
            diff_mean, last_mean = curr_mean-last_mean, curr_mean
            print('\r[Epoch %i] 100.0%% mean %.5f diff %.5f %s' %
                (epoch, curr_mean, diff_mean, str(datetime.now())[11:19])) 
    
    def predict(self, input):
        return self.sess.run(self.output_op, feed_dict={self.input: input})

