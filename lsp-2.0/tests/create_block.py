import tensorflow as tf
from tensorflow_extensions import *

# means the function should return a tensor with shape (1, 224, 224, 3), dtype float32 and device cpu
@tensor_assert(output={"shape": (1, 224, 224, 3), "dtype": "float32", "device": "cpu"})
def create_img_input_tensor():
    return tf.zeros((1, 224, 224, 3), dtype=tf.float32)

x = create_img_input_tensor()

@conv2d_assert(filters=64, kernel_size=7, strides=2, padding="same")
def conv2d(input):
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same")(input)
    return x

# means the function should keep the shape, dtype and device of the input tensor
@scalar_op_assert
def relu(input):
    x = tf.nn.relu(input)
    return x

# means the function should accept a tensor with shape (1, 224, 224, 3), dtype float32 and device cpu
# and return a tensor with shape (1, 112, 112, 64), dtype float32 and device cpu
@block_assert(
    inputs=(
        {"shape": (1, 224, 224, 3), "dtype": "float32", "device": "cpu"},
    ),
    output={"shape": (1, 112, 112, 32), "dtype": "float32", "device": "cpu"}
)
def resnet(input):
    x = conv2d(input)
    x = relu(x)
    return x


pred_y = resnet(x)
