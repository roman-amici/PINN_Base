import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import Progbar
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


class FeedForward(Layer):

    def __init__(self, lower_bound, upper_bound, layers, activation="tanh", **kwargs):

        self.lower_bound = tf.convert_to_tensor(lower_bound, dtype=tf.float32)
        self.upper_bound = tf.convert_to_tensor(upper_bound, dtype=tf.float32)

        self.layers = [
            Dense(layers[i], activation=activation)
            for i in range(1, len(layers)-1)]

        self.layers.append(Dense(layers[-1], activation="linear"))

        super().__init__(**kwargs)

    def call(self, X):

        H = 2.0 * (X - self.lower_bound) / \
            (self.upper_bound - self.lower_bound) - 1.0

        for l in self.layers:
            H = l(H)

        return H


class PINN_Base:
    # TODO: integrate dtype. Will default to float32 for now.
    def __init__(self, lower_bound, upper_bound, layers, use_differential_points=True):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.layers = layers

        self.use_differential_points = use_differential_points

        self._init()

    def _init(self):
        self._init_layers()

    def _init_layers(self):
        self.NN = FeedForward(self.lower_bound, self.upper_bound, self.layers)
        self.mse = MeanSquaredError()

    def _forward(self, X):
        return self.NN(X)

    def _loss(self, X, U, X_df=None):

        with tf.GradientTape(persistent=True) as g:
            # Add the input variables to the tape so that we can take
            # derivatives from them in the residual.
            g.watch(X)
            if self.use_differential_points:
                g.watch(X_df)

            U_hat = self._forward(X)
            mse = self.mse(U, U_hat)
            loss_residual = self.mse(
                0, self._residual_collocation(X, U_hat, g, U))

            if self.use_differential_points:
                U_df = self._forward(X_df)
                loss_residual_differential = self.mse(
                    0, self._residual_differential(X_df, U_df, g))

                # TODO: Move this off the tape?
                loss = mse + loss_residual + loss_residual_differential
            else:
                loss_residual_differential = None
                loss = mse + loss_residual

        del g  # Since we use persistent, we need to clean this up explicitly
        return loss, [mse, loss_residual, loss_residual_differential]

    def _residual(self, u, x, gradient_tape, u_true=None):

        # Fill this in with your differential equation
        return tf.constant(0.0)

    def _residual_collocation(self, X, U_hat, gradient_tape, U_true):
        return self._residual(U_hat, X, gradient_tape, U_true)

    def _residual_differential(self, X, U_hat, gradient_tape):
        return self._residual(U_hat, X, gradient_tape)

    def _training_loop_optimizer(self, X, U, X_df, epochs, optimizer):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        U = tf.convert_to_tensor(U, dtype=tf.float32)

        if self.use_differential_points:
            X_df = tf.convert_to_tensor(X_df, dtype=tf.float32)
        else:
            X_df = None

        progbar = Progbar(epochs, stateful_metrics=["loss"])
        for i in range(epochs):
            with tf.GradientTape() as param_tape:
                loss, _ = self._loss(X, U, X_df)
            params = self.get_trainable_parameters()
            grads = param_tape.gradient(
                loss, params)
            optimizer.apply_gradients(zip(grads, params))

            progbar.update(i+1, [("loss", loss)])

    def get_trainable_parameters(self):
        return self.NN.trainable_weights

    def get_trainable_parameters_layered(self):
        weights = self.NN.trainable_weights
        weights, biases = [], []
        for i, w in weights:
            if i % 2 == 0:
                weights.append(w)
            else:
                biases.append(w)

        return weights, biases

    def train_Adam(self, X, U, X_df, epochs, **kwargs):
        adam = Adam(**kwargs)
        self._training_loop_optimizer(X, U, X_df, epochs, adam)
