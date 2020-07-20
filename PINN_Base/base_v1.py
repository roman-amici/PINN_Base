import tensorflow as tf
import importlib
import numpy as np
from tensorflow.contrib.opt import ScipyOptimizerInterface
from tensorflow.keras.utils import Progbar
import PINN_Base.util as util
from sklearn.utils import shuffle

from typing import List


class PINN_Base:
    def __init__(self,
                 lower_bound: List[float],
                 upper_bound: List[float],
                 layers: List[int],
                 dtype=tf.float32,
                 use_differential_points=True,
                 use_collocation_residual=True,
                 optimizer_kwargs={},
                 session_config=None,
                 df_multiplier=1.0,
                 add_grad_ops=False):

        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)

        self.layers = layers

        self.dtype = dtype
        self.use_differential_points = use_differential_points
        # When using differential points, this decides whether you want to
        # evaluate the residual on the collocation points as well as the differential points.
        # If true, you will have a three term loss instead of a 2 term loss.
        self.use_collocation_residual = use_collocation_residual
        self.optimizer_kwargs = optimizer_kwargs
        self.session_config = session_config
        self.add_grad_ops = add_grad_ops

        self.df_multiplier = df_multiplier

        self.graph = tf.Graph()
        self._build_graph()

    def _build_graph(self):

        with self.graph.as_default():

            self._init_placeholders()
            self._init_params()

            self.U_hat = self._forward(self.X)

            if self.use_differential_points:
                self.U_hat_df = self._forward(self.X_df)
            else:
                self.U_hat_df = None

            self.loss = self._loss(self.U_hat, self.U_hat_df)

            if self.add_grad_ops:
                self._init_grad_ops()

            self._init_optimizers()

            self.init = tf.global_variables_initializer()
            self.graph.finalize()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def _init_session(self):

        if self.session_config is not None:
            self.sess = tf.Session(
                graph=self.graph, config=self.session_config)
            self.sess.run(self.init)

    def _init_placeholders(self):

        self.X = tf.placeholder(self.dtype, shape=[None, self.get_input_dim()])
        if self.use_differential_points:
            self.X_df = tf.placeholder(
                self.dtype, shape=[None, self.get_input_dim()])
        self.U = tf.placeholder(
            self.dtype, shape=[None, self.get_output_dim()])

    def _init_params(self):

        self.weights, self.biases = self._init_NN(self.layers)

    def _init_session(self):

        if self.session_config is not None:
            self.sess = tf.Session(
                graph=self.graph, config=self.session_config)

        else:
            self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def _forward(self, X):

        U, activations = self._NN(X, self.weights, self.biases)

        # By convention we only store generic values for a single forward-path
        if X == self.X:
            self.activations = activations

        return U

    def _init_optimizers(self):
        self.optimizer_BFGS = ScipyOptimizerInterface(
            self.loss,
            method='L-BFGS-B',
            options={'maxiter': 50000,
                     'maxfun': 50000,
                     'maxcor': 50,
                     'maxls': 50,
                     'gtol': 1.0 * np.finfo(float).eps,
                     'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(
            **self.optimizer_kwargs).minimize(self.loss)

    def _loss(self, U_hat, U_hat_df):

        self.mse = tf.reduce_mean(tf.square(self.U - U_hat))
        self.loss_residual = tf.reduce_mean(
            tf.square(self._residual_collocation(U_hat)))

        if self.use_differential_points:
            self.loss_residual_differential = tf.reduce_mean(
                tf.square(self._residual_differential(U_hat_df)))

            if self.use_collocation_residual:
                return self.mse + self.loss_residual + self.df_multiplier * self.loss_residual_differential
            else:
                return self.mse + self.df_multiplier * self.loss_residual_differential
        else:
            return self.mse + self.df_multiplier * self.loss_residual

    def _residual(self, u, x, u_true=None):

        # Fill this in with your differential equation
        return 0.0

    def _residual_collocation(self, U_hat):
        return self._residual(U_hat, self.X, self.U)

    def _residual_differential(self, U_hat_df):
        return self._residual(U_hat_df, self.X_df)

    def _NN(self, X, weights, biases):

        activations = []

        H = 2.0 * (X - self.lower_bound) / \
            (self.upper_bound - self.lower_bound) - 1.0
        activations.append(H)

        for l in range(len(weights) - 1):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
            activations.append(H)

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)

        return Y, activations

    def _xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        stddev = np.sqrt(2 / (in_dim + out_dim))

        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=stddev, dtype=self.dtype), dtype=self.dtype)

    def _init_NN(self, layers):
        weights = []
        biases = []
        for l in range(len(layers) - 1):
            W = self._xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(
                tf.zeros([1, layers[l + 1]], dtype=self.dtype), dtype=self.dtype)
            weights.append(W)
            biases.append(b)

        return weights, biases

    def _init_grad_ops(self):
        # Add ops to get the gradient outside of an optimizer
        # Also add ops useful for computing the full hessian or a hessian approximation

        all_params = self.get_all_weight_variables()
        param_array = []
        for param_list in all_params:
            for layer in param_list:
                param_array.append(layer)

        # Can't concat yet since the flattened array is not part of the computation
        # of the loss.
        grads = tf.gradients(self.loss, param_array)
        grads_flat = []
        for grad in grads:
            grads_flat.append(tf.reshape(grad, [-1]))

        self.grads_flat = tf.concat(grads_flat, axis=0)

        self.hessian_vector = tf.placeholder(
            self.dtype, shape=self.grads_flat.shape)

        prod = tf.reduce_sum(self.grads_flat * self.hessian_vector)
        self.hessian_matvec = tf.gradients(prod, param_array)

    def get_input_dim(self):
        return self.layers[0]

    def get_output_dim(self):
        return self.layers[-1]

    def reset_session(self):
        self.sess.close()
        self._init_session()

    def cleanup(self):
        del self.graph
        self.sess.close()

    def get_all_weights(self):
        # return all weight parameters as a list
        # here to be changed by derived classes
        return self.get_weights()

    def get_all_weight_variables(self):
        # Get a list (with the same shape as in get_all_weights)
        # with the tf objects corresponding to all weights
        return [self.weights, self.biases]

    def get_weights(self):
        return self.sess.run([self.weights, self.biases])

    def get_loss(self, X):
        return self.sess.run(self.loss, {self.X: X})

    def get_loss_collocation(self, X):
        return self.sess.run(self.loss)

    def get_loss_residual(self, X):
        return self.sess.run(self.loss_residual)

    def get_activations(self, X, layer=None):
        if layer:
            return self.sess.run(self.activations[layer], {self.X: X})
        else:
            return self.sess.run(self.activations, {self.X: X})

    def _size_of_variable_list(self, variable_list):
        l = self.sess.run(variable_list)
        return np.sum([
            v.size for v in l
        ])

    def _count_params(self):
        params_weights = self._size_of_variable_list(self.weights)
        params_biases = self._size_of_variable_list(self.biases)

        return params_weights + params_biases

    def get_architecture_description(self):
        params = self._count_params()
        return {
            "arch_name": "base",
            "n_params": params,
            "shape": self.layers[:],
            "dtype": "float32" if self.dtype == tf.float32 else "float64"
        }

    def get_version(self):
        return tf.__version__

    def train_BFGS(self, X, U, X_df=None, print_loss=True, custom_fetches=None):
        # TODO: Support printing loss and doing custom fetching...

        if self.use_differential_points:
            feed_dict = {self.X: X, self.U: U, self.X_df: X_df}
        else:
            feed_dict = {self.X: X, self.U: U}

        if print_loss:
            self.optimizer_BFGS.minimize(
                self.sess, feed_dict, fetches=[self.loss], loss_callback=util.bfgs_callback)
        elif custom_fetches is not None:
            array, callback = util.make_fetches_callback()

            self.optimizer_BFGS.minimize(
                self.sess, feed_dict, fetches=custom_fetches, loss_callback=callback)

            return array
        else:
            self.optimizer_BFGS.minimize(
                self.sess, feed_dict)

    def train_Adam(self, X, U, X_df=None, n_iter=2000):

        if self.use_differential_points:
            feed_dict = {self.X: X, self.U: U, self.X_df: X_df}
        else:
            feed_dict = {self.X: X, self.U: U}

        progbar = Progbar(n_iter)
        for i in range(n_iter):
            _, loss = self.sess.run(
                [self.optimizer_Adam, self.loss], feed_dict)

            progbar.update(i + 1, [("loss", loss)])

    def train_Adam_batched(self, X, U, X_df=None, batch_size=128, epochs=10):
        self._train_stochastic_optimizer(
            self.optimizer_Adam, X, U, X_df, batch_size, epochs)

    def _train_stochastic_optimizer(self, optimizer_opp, X, U, X_df=None, batch_size=128, epochs=10):

        if self.use_differential_points:
            assert(X_df is not None)

            assert(X_df.shape[0] >= X.shape[0])

        progbar = Progbar(epochs, stateful_metrics=["loss_full"])
        for epoch in range(epochs):

            X_s, U_s = shuffle(X, U)

            if X_df is not None:
                X_df_s = shuffle(X_df)
                dataset_size = X_df.shape[0]
            else:
                dataset_size = X.shape[0]

            b_c = 0
            for b in range(0, dataset_size, batch_size):

                if X_df is not None:
                    b_c_last = b_c
                    b_c = b % X_s.shape[0]

                    if b_c_last > b_c:
                        X_s, U_s = shuffle(X, U)
                    X_b = X_s[b_c:(b_c + batch_size), :]
                    U_b = U_s[b_c:(b_c + batch_size), :]
                    X_df_b = X_df_s[b:(b + batch_size), :]
                    feed_dict = {self.X: X_b, self.U: U_b, self.X_df: X_df_b}
                else:
                    X_b = X_s[b:(b + batch_size), :]
                    U_b = U_s[b:(b + batch_size), :]
                    feed_dict = {self.X: X_b, self.U: U_b}

                _, loss = self.sess.run(
                    [optimizer_opp, self.loss], feed_dict)

            if X_df is not None:
                feed_dict = {self.X: X, self.U: U, self.X_df: X_df}
            else:
                feed_dict = {self.X: X, self.U: U}

            progbar.update(epoch + 1, [("loss", loss)])

    def predict(self, X):
        return self.sess.run(self.U_hat, {self.X: X})

    def get_hessian_OPG(self, X, U, X_df):

        if self.use_differential_points:
            feed_dict = {self.X: X, self.U: U, self.X_df: X_df}
        else:
            feed_dict = {self.X: X, self.U: U}

        grads = self.sess.run(self.grads_flat, feed_dict)
        return np.outer(grads, grads)

    def get_hessian_matvec(self, v, X, U, X_df):

        if self.use_differential_points:
            feed_dict = {self.hessian_vector: v,
                         self.X: X, self.U: U, self.X_df: X_df}
        else:
            feed_dict = {
                self.hessian_vector: v,
                self.X: X, self.U: U}

        h_row = self.sess.run(self.hessian_matvec, feed_dict)

        # h_row is a list, we want to return a vector
        return util.unwrap(h_row)

    def get_hessian(self, X, U, X_df):
        print(
            "Warning, trying to calculate the full Hessian is infeasible for large networks!")

        if self.use_differential_points:
            feed_dict = {self.X: X, self.U: U, self.X_df: X_df}
        else:
            feed_dict = {self.X: X, self.U: U}

        # We use repeated runs to avoid adding gradient ops for every
        # element of the hessian
        n = int(self.grads_flat.shape[0])
        H = np.empty((n, n))
        progbar = Progbar(n)
        for i in range(n):
            vec = np.zeros(n, dtype=np.float32)
            vec[i] = 1.0
            feed_dict[self.hessian_vector] = vec
            h_row = self.sess.run(self.hessian_matvec, feed_dict)
            h_row = util.unwrap(h_row)
            H[i, :] = h_row[:]
            progbar.update(i + 1)

        # Explicitly diagonalize so that e.g. eigenvalues are always real
        for i in range(n):
            for j in range(i + 1, n):
                H[j, i] = H[i, j]
        return H
