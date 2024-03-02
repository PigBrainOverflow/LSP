from tensorflow_extensions import *
import tensorflow as tf

@tensor_assert(output={"shape": (224, 224, 3), "dtype": "float32", "device": "cpu"})
def create_img_input_tensor():
    return tf.zeros((224, 224, 3), dtype=tf.float32)

x = create_img_input_tensor()