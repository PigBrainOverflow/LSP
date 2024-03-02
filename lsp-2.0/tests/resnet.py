from tensorflow_extensions import *
import tensorflow as tf


@tensor_assert(output={"shape": (100, 224, 224, 3), "dtype": "float32", "device": "cpu"})
def create_tensor():
    return tf.zeros((100, 224, 224, 3), dtype=tf.float32)

@scalar_op_assert
def relu(input):
    return tf.nn.relu(input)

@conv2d_assert(filters=64, kernel_size=7, strides=2, padding="same")
def conv_1(input):
    return tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="same")(input)

@pooling2d_assert(pool_size=3, strides=2, padding="same")
def max_pooling_1(input):
    return tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(input)

@conv2d_assert(filters=64, kernel_size=3, strides=1, padding="same")
def conv_2(input):
    return tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(input)

@tensor_assert(output={"shape": (100, 1000), "dtype": "float32", "device": "cpu"})
def dense_1(input):
    return tf.keras.layers.Dense(1000)(input)

@block_assert(
    inputs=(
        {"shape": (100, 224, 224, 3), "dtype": "float32", "device": "cpu"},
    ),
    output={"shape": (100, 1000), "dtype": "float32", "device": "cpu"}
)
def resnet18(input):

    x = conv_2(max_pooling_1(conv_1(input)))

    # identity block
    x = relu(x + conv_2(conv_2(x)))

    # the correct version is x = relu(dense_1(x))
    x = relu(x + dense_1(x))

    return x

pred_y = resnet18(create_tensor())

