import tensorflow as tf
import numpy as np
from keras.models import model_from_config
from keras.optimizers import optimizer_from_config, get



def clone_model(model, custom_objects={}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone


def clone_optimizer(optimizer):
    if type(optimizer) is str:
        return get(optimizer)
    params = dict([(k, v) for k, v in optimizer.get_config().items()])
    config = {
        'class_name': optimizer.__class__.__name__,
        'config': params,
    }
    clone = optimizer_from_config(config)
    return clone


def huber_loss(y_true, y_pred, clip_value):
    assert clip_value > 0.0

    x = y_true - y_pred
    if np.isinf(clip_value):
        return 0.5 * tf.square(x)

    condition = tf.abs(x) < clip_value
    squared_loss = 0.5 * tf.square(x)
    linear_loss = clip_value * (tf.abs(x) - 0.5 * clip_value)
    return tf.select(condition, squared_loss, linear_loss)
