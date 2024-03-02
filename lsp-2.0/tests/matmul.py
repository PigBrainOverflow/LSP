from tensorflow_extensions import *
import tensorflow as tf

@tensor_assert(output={"shape": (224, 3, 4), "dtype": "float32", "device": "cpu"})
def create_tensor_x():
    return tf.zeros((224, 224, 3), dtype=tf.float32)

@tensor_assert(output={"shape": (224, 4, 5), "dtype": "float32", "device": "cpu"})
def create_tensor_y():
    return tf.zeros((224, 224, 2), dtype=tf.float32)

@tensor_assert(output={"shape": (224, 5, 5), "dtype": "float32", "device": "cpu"})
def create_tensor_z():
    return tf.zeros((224, 224, 5), dtype=tf.float32)

x = create_tensor_x()
y = create_tensor_y()
z = create_tensor_z()
res = z + x @ y
