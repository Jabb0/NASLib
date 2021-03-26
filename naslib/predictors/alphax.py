import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

from naslib.predictors.utils.encodings import encode
from naslib.predictors.predictor import Predictor

"""
This is a preliminary implementation of the predictor from AlphaX [Wang et al. 2019]
Note: it is still not a fully accurate implementation. The architecture is correct,
but the output, loss function, optimizer, etc need to be switched over to the AlphaX
version.
"""


def mle_loss(y_true, y_pred):
    # Minimum likelihood estimate loss function
    mean = tf.slice(y_pred, [0, 0], [-1, 1])
    var = tf.slice(y_pred, [0, 1], [-1, 1])
    return 0.5 * tf.log(2*np.pi*var) + tf.square(y_true - mean) / (2*var)


def mape_loss(y_true, y_pred):
    # Minimum absolute percentage error loss function
    lower_bound = 4.5
    fraction = tf.math.divide(tf.subtract(y_pred, lower_bound), \
        tf.subtract(y_true, lower_bound))
    return tf.abs(tf.subtract(fraction, 1))


class AlphaXPredictor(Predictor):

    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201', hpo_wrapper=False):
        self.encoding_type = encoding_type
        self.ss_type = ss_type
    
    def get_model(self,
                  input_dims,
                  num_layers,
                  layer_width,
                  loss,
                  regularization):
        input_layer = keras.layers.Input(input_dims)
        model = keras.models.Sequential()
        
        model.add(keras.layers.Dense(512, activation='relu', use_bias=True, kernel_initializer='RandomUniform',
                                     bias_initializer='zeros'))
        model.add(
            keras.layers.Dense(2048, activation='relu', use_bias=True, kernel_initializer='RandomUniform', bias_initializer='zeros'))
        model.add(
            keras.layers.Dense(2048, activation='relu', use_bias=True, kernel_initializer='RandomUniform', bias_initializer='zeros'))
        model.add(
            keras.layers.Dense(512, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
        #model.add(keras.layers.Dense(1, activation='sigmoid', use_bias=True))


        model = model(input_layer)
        if loss == 'mle':
            mean = keras.layers.Dense(1)(model)
            var = keras.layers.Dense(1)(model)
            var = keras.layers.Activation(tf.math.softplus)(var)
            output = keras.layers.concatenate([mean, var])
        else:
            if regularization == 0:
                output = keras.layers.Dense(1)(model)
            else:
                reg = keras.regularizers.l1(regularization)
                output = keras.layers.Dense(1, kernel_regularizer=reg)(model)

        dense_net = keras.models.Model(inputs=input_layer, outputs=output)
        return dense_net
    
    def fit(self, xtrain, ytrain, train_info=None,
            num_layers=20,
            layer_width=20,
            loss='mae',
            epochs=500,
            batch_size=32,
            lr=.001,
            verbose=0,
            regularization=0.2):
        
        xtrain = np.array([encode(arch, encoding_type=self.encoding_type, 
                                  ss_type=self.ss_type) for arch in xtrain])
        ytrain = np.array(ytrain)

        if loss == 'mle':
            loss_fn = mle_loss
        elif loss == 'mape':
            loss_fn = mape_loss
        else:
            loss_fn = 'mae'
            
        self.model = self.get_model((xtrain.shape[1],),
                                    loss=loss_fn,
                                    num_layers=num_layers,
                                    layer_width=layer_width,
                                    regularization=regularization)
        optimizer = keras.optimizers.Adam(lr=lr, beta_1=.9, beta_2=.99)

        self.model.compile(optimizer=optimizer, loss=loss_fn)
        
        self.model.fit(xtrain, ytrain, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=verbose)

        train_pred = np.squeeze(self.model.predict(xtrain))
        train_error = np.mean(abs(train_pred-ytrain))
        return train_error
    
    def query(self, xtest, info=None):
        xtest = np.array([encode(arch, encoding_type=self.encoding_type, 
                                 ss_type=self.ss_type) for arch in xtest])
        xtest = np.array(xtest)
        pred = self.model.predict(xtest)
        return np.squeeze(pred)
