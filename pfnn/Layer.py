import tensorflow as tf
import numpy as np


class Layer(object):
    def __init__(self):
        self.params = None

    def save(self, sess, database={}, prefix=''):
        if not self.params is None:
            for param in self.params:
                # if param.name in sess.run(tf.report_uninitialized_variables()):
                #     sess.run()
                if not tf.is_variable_initialized(param) is None:
                    sess.run(param.initializer)
                database[prefix + param.name] = np.array(Layer.get_param_value(param, sess))
        return database

    def load(self, sess, database, prefix='', mapping_dict=None):
        if not self.params is None:
            for param in self.params:
                # print(param.name)
                if not mapping_dict is None:
                    assign_op = param.assign(database[mapping_dict[prefix+param.name]])
                else:
                    assign_op = param.assign(database[prefix+param.name])
                sess.run(assign_op)

    def cost(self):
        return 0

    @staticmethod
    def get_param_value(param, sess):
        return sess.run(param)