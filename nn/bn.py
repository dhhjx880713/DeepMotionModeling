# encoding: UTF-8

import tensorflow as tf
import numpy as np


def sqrt_init(shape):
    value = (1 / tf.sqrt(2)) * tf.ones(shape)
    return value


def sanitizedInitGet(init):
    if init in ["sqrt_init"]:
        return sqrt_init
    else:
        return tf.initializers.get(init)
    
    
def sanitizedInitSer(init):
    if init in [sqrt_init]:
        return "sqrt_init"
    else:
        return tf.initializers.serialize(init)
