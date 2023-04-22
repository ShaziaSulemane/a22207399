import tensorflow as tf
import tensorflow_hub as hub


def nelu(x):
    """
    A Relu function that does not get rid of the negative values,
    since the input data has (-1, -1) for points that are not in the input image
    some way of keeping those values negative is needed since (0, 0) is a valid coordinate
    This way any negative value is -1, and all positives follow the normal relu outputs
    :param x: float Tensor to perform activation.
    :return: value of x, -1 if negative, equal if positive
    """
    if x < 0:
        return -1
    else:
        return x



def create_model(model_url, target_size, num_coordinates, activation_function=nelu):
    """
    Create a model using the tensorflow hub. The final model has a feature_extractor_layer where the model from
    the model_url is and an output dense layer of (num_coordinates, 2)
    :param activation_function: the name of the activation function to use
    :param model_url: the model url from tensorflow hub
    :param target_size: input shape of the tensorflow hub model
    :param num_coordinates: how many coordinates in each image from the dataset
    :return: the model with a final output dense layer
    """
    # download the pre-trained model and save it as a keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,
                                             name="Feature_Extraction_Layer",
                                             input_shape=target_size + (3,))  # freeze the already learned patterns
    # create our own model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.Dense((num_coordinates, 2), activation=activation_function, name="output_layer")
    ])

    return model

