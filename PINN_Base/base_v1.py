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
                 df_multiplier=1.0,
                 dtype=tf.float32,
                 use_differential_points=True,
                 use_collocation_residual=True,
                 use_dynamic_learning_rate=False,
                 optimizer_kwargs={},
                 session_config=None,
                 add_grad_ops=False):
        '''
        Inherit from this class to construct a Physics informed neural network (PINN)
        The computation graph is constructed at initialization and then finalized.

        Parameters:
            lower_bound (List[float]) : Lower bound on the input domain for each coordinate
            upper_bound (List[float]) : Upper bound on the input domain for each coordinate
            layers (List[int]) : List describing the input/output domain of MLP used by the PINN.
                List should have the form [ input_dimension, layer_width*, output_dimension]
                where layer_width* is 0 or more numbers represting the width of each fully connected layer.
                For instance, [2,20,20,20,20,1] is an MLP with a 2d input domain, 4 fully connected layers of
                width 20 and a scalar output domain.
            df_multiplier (float) : Value which multiplies the PINN residual portion of the loss.
                < 1.0 means that the effect of the residual will be reduced
                > 1.0 means that the effect of the residual will be magnified
            dtype (tf.dtype) : Data type to use for *all* computations.
                Warning! tf.float64 is often a bit more accurate but much slower!
            use_differential_points (bool) : Whether to use a separate set of differential points (X_df)
                when calculating the residual. Setting this to false switches us from the "boundary-value" paradigm
                to the "noisy-sensor" paradigm.
            use_collocation_residual (bool) : Determines if we have a 2-term or 3-term loss (see _loss for more details)
            use_dynamic_learning_rate (bool) : When using a first order optimizer, determines whether the learning rate is changable after graph compilation
            optmizer_kwargs (dict) : Args passed to first order optimizer by default
            session_config (dict) : Arguments passed to the tf.session on creation.
                This may be used to force device placement or limit the number of threads that may be used.
            add_grad_ops (bool) : If true, adds operations to compute the gradient outside of an optimizer.
                See _init_grad_ops for more details
        '''

        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)

        self.layers = layers

        self.dtype = dtype
        self.use_differential_points = use_differential_points

        self.use_collocation_residual = use_collocation_residual
        self.optimizer_kwargs = optimizer_kwargs
        self.session_config = session_config
        self.add_grad_ops = add_grad_ops
        self.use_dynamic_learning_rate = use_dynamic_learning_rate

        self.df_multiplier = df_multiplier

        self.graph = tf.Graph()
        self._build_graph()

    def _build_graph(self):
        '''
        Builds the full computation graph.
        Each _method is meant to be an integration point modifiable by subclasses
        '''

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

    def _init_placeholders(self):
        '''Initialize training inputs here'''

        self.X = tf.placeholder(self.dtype, shape=[None, self.get_input_dim()])
        if self.use_differential_points:
            self.X_df = tf.placeholder(
                self.dtype, shape=[None, self.get_input_dim()])
        self.U = tf.placeholder(
            self.dtype, shape=[None, self.get_output_dim()])

        if self.use_dynamic_learning_rate:
            self.learning_rate = tf.placeholder(self.dtype, shape=())
        else:
            self.learning_rate = None

    def _init_params(self):
        '''Initialize trainable parameters here'''

        self.weights, self.biases = self._init_NN(self.layers)

    def _init_session(self):

        if self.session_config is not None:
            self.sess = tf.Session(
                graph=self.graph, config=self.session_config)

        else:
            self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def _forward(self, X):
        '''
        Computes U = F(X)
        '''

        U, activations = self._NN(X, self.weights, self.biases)

        # By convention we only store intermediate for a single forward-pass
        if X == self.X:
            self.activations = activations

        return U

    def _init_optimizers(self):
        '''
        Initialize optimizers
        By default LBFGS-B and Adam are initialized.
        '''

        self.optimizer_BFGS = ScipyOptimizerInterface(
            self.loss,
            method='L-BFGS-B',
            options={'maxiter': 50000,
                     'maxfun': 50000,
                     'maxcor': 50,
                     'maxls': 50,
                     'gtol': 1.0 * np.finfo(float).eps,
                     'ftol': 1.0 * np.finfo(float).eps})

        if self.learning_rate is not None:
            self.optimizer_Adam = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)
        else:
            self.optimizer_Adam = tf.train.AdamOptimizer(
                **self.optimizer_kwargs).minimize(self.loss)

    def _loss(self, U_hat, U_hat_df):
        '''
            Computes the loss
        '''

        # Fit a given set of points with known values
        self.mse = tf.reduce_mean(tf.square(self.U - U_hat))
        # The PINN residual for the same points as the mse
        self.loss_residual = tf.reduce_mean(
            tf.square(self._residual_collocation(U_hat)))

        if self.use_differential_points:
            # The PINN residual for a separate set of points, X_df
            # which need to have known values of U
            self.loss_residual_differential = tf.reduce_mean(
                tf.square(self._residual_differential(U_hat_df)))

            # The two sets of points gives us two residuals.
            # It can be beneficial in some cases to use both of them.
            if self.use_collocation_residual:
                return self.mse + self.loss_residual + self.df_multiplier * self.loss_residual_differential
            else:
                return self.mse + self.df_multiplier * self.loss_residual_differential
        else:
            return self.mse + self.df_multiplier * self.loss_residual

    def _residual(self, u, x, u_true=None):
        '''
            Computes the PINN residual.
            Fill in this mehtod to create a PINN for a *particular*
            differential equation.

            Parameters:
                u (tf.tensor) : Predicted value of the differential equation
                x (tf.tensor) : Input to the differential equation
                u_true (Optional[tf.tensor]) : In some cases it can be helpful to
                    use the true value of u (if its known) in the differential residual
                    rather than u which is the prediction of the neural network.
        '''

        # Fill this in with your differential equation
        return 0.0

    def _residual_collocation(self, U_hat):
        '''
            Residual for X, U with U known. Note that this is
            ignored if use_differential_points=True and use_collocation_residual=False
        '''
        return self._residual(U_hat, self.X, self.U)

    def _residual_differential(self, U_hat_df):
        '''
            Residual for X_df with U unknown.
            Ignored if use_differential_points=False
        '''
        return self._residual(U_hat_df, self.X_df)

    def _NN(self, X, weights, biases):
        '''
            A simple MLP neural network.
            Original code based on (Raissi, 2018)
        '''
        activations = []

        # Scale the inputs to be between -1 and 1 to help with optimization
        H = 2.0 * (X - self.lower_bound) / \
            (self.upper_bound - self.lower_bound) - 1.0

        # Save off the activations at each layer for visualization, etc
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
        '''
        Xavier Initialization for layer of give size.
        Code based on (Raissi, 2018)
        '''

        in_dim = size[0]
        out_dim = size[1]
        stddev = np.sqrt(2 / (in_dim + out_dim))

        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=stddev, dtype=self.dtype), dtype=self.dtype)

    def _layer_initializer(self, size):
        return self._xavier_init(size)

    def _bias_initializer(self, width):
        return tf.Variable(
            tf.zeros([1, width], dtype=self.dtype), dtype=self.dtype)

    def _init_NN(self, layers: List[float]):
        '''
        Initialize the weights and biases for the MLP with given structure

        Parameters:
            layers (list[float]) : See the same param in __init__
        '''

        weights = []
        biases = []
        for l in range(len(layers) - 1):
            W = self._layer_initializer([layers[l], layers[l + 1]])
            b = self._bias_initializer(layers[l + 1])
            weights.append(W)
            biases.append(b)

        return weights, biases

    def _init_grad_ops(self):
        '''
            Adds operations for querying the gradient of the loss with respect to the trainable
            parameters outside of an optimization context.
            Also adds operations to efficiently compute Hv where H is the hessian and v a given vector.
        '''

        # Parameters as a list of list of lists, we need a list of lists instead
        all_params = self.get_all_weight_variables()
        param_array = []
        for param_list in all_params:
            for layer in param_list:
                param_array.append(layer)

        # Can't concat the list yet since the flattened array is not part of the
        # computation of the loss. So we first compute the gradients and then flatten.
        grads = tf.gradients(self.loss, param_array)
        grads_flat = []
        for grad in grads:
            grads_flat.append(tf.reshape(grad, [-1]))

        self.grads_flat = tf.concat(grads_flat, axis=0)

        # v in Hv
        self.hessian_vector = tf.placeholder(
            self.dtype, shape=self.grads_flat.shape)

        # As long as v is idependent of L, grad ((grad L).T v) = Hv
        prod = tf.reduce_sum(self.grads_flat * self.hessian_vector)
        self.hessian_matvec = tf.gradients(prod, param_array)

    def get_input_dim(self):
        '''Dimension of the domain of the differential equation'''
        return self.layers[0]

    def get_output_dim(self):
        '''
        Dimension of the range of the differential equation
        Note that most this code has only been tested for 1d output dimensions
        '''
        return self.layers[-1]

    def reset_session(self):
        '''
        Reset the model without rebuilding the graph.
        This is faster for multiple trials with identical architecture.
        '''
        self.sess.close()
        self._init_session()

    def cleanup(self):
        '''
        Not sure if this is actually needed. I believe
        Tensorflow has gotten better about not leaking memory at this point.
        '''
        del self.graph
        self.sess.close()

    def get_all_weights(self):
        '''
        Get all trainable parameters as a list of list of lists.
        This should return the results of tf.sess.run, not the variable ops themselves.
        Inheriting classes should modify this function and not get_weights()
        to return additional parameters.
        '''
        return self.get_weights()

    def get_all_weight_variables(self):
        '''
        Returns a list (with the same shape as in get_all_weights)
        with the tf.Variable objects corresponding to all trainable parameters
        '''
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
        '''
            Used to count the total number of parameters in a list of tf.Variable objects
        '''
        l = self.sess.run(variable_list)
        return np.sum([
            v.size for v in l
        ])

    def _count_params(self):
        '''
            The total number of parameters used by the model.
        '''
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
        '''
            Train the model to completion using L-BFGS-B

            Parameters:
                X (np.ndarray) : (N,d_in) array of domain points
                U (np.ndarray) : (N,d_out) array of solution points such that U = F(X)
                X_df (Optional[np.ndarray]) : (M,d_in) array of domain points where U is 
                    unknown but the PINN residual should still be evaluated
                print_loss (bool) : Whether to print the loss to stdout during training
                custom_fetches (List) : Ops from the computation graph to fetch at each training step.

            Returns:
                If custom_fetches were supplied, the fetched values will be returned. 
                Otherwise nothing will be returned. 
        '''

        # TODO: Support printing loss and doing custom fetching at the same time

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

    def train_Adam(self, X: np.ndarray, U: np.ndarray, X_df=None, epochs=2000, learning_rate=1e-3):
        '''
            Train using Full-Batch Adam for the given number of iterations

            Parameters:
                X (np.ndarray) : (N,d_in) array of domain points
                U (np.ndarray) : (N,d_out) array of solution points such that U = F(X)
                X_df (Optional[np.ndarray]) : (M,d_in) array of domain points where U is 
                    unknown but the PINN residual should still be evaluated.
                epochs (int) : Number of epochs to train for
                learning_rate (float) : If use_dynamic_learning_rate=True, this will 
                    be the learning rate used by the optimizer
        '''

        if self.use_differential_points:
            feed_dict = {self.X: X, self.U: U, self.X_df: X_df}
        else:
            feed_dict = {self.X: X, self.U: U}

        if self.learning_rate is not None:
            feed_dict[self.learning_rate] = learning_rate

        progbar = Progbar(epochs)
        for i in range(epochs):
            _, loss = self.sess.run(
                [self.optimizer_Adam, self.loss], feed_dict)

            progbar.update(i + 1, [("loss", loss)])

    def train_Adam_batched(self, X, U, X_df=None, batch_size=128, epochs=10):
        '''
            Train using Mini-Batch Adam for the given number of iterations

            Parameters:
                X (np.ndarray) : (N,d_in) array of domain points
                U (np.ndarray) : (N,d_out) array of solution points such that U = F(X)
                X_df (Optional[np.ndarray]) : (M,d_in) array of domain points where U is 
                    unknown but the PINN residual should still be evaluated.
                epochs (int) : Number of epochs to train for
                batch_size (int) : Mini-Batch size for stochastic training
        '''
        # TODO: Integrate with dynamic learning rate

        self._train_stochastic_optimizer(
            self.optimizer_Adam, X, U, X_df, batch_size, epochs)

    def _train_stochastic_optimizer(self, optimizer_opp, X, U, X_df=None, batch_size=128, epochs=10):
        '''
            Generic custom training loop for stochastic optimizers. 
            Replace optimizer_opp with e.g. RMSProp.minimize() for a different
            stochastic optimizer.
        '''

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

                    # X and X_df are typically different sizes,
                    # so we shuffle them at different times
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

    def get_hessian_matvec(self, v, X, U, X_df):
        '''
            Get the result of Hv for the given vector v
        '''

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
        '''
            Get the full hessian by repeated calls to Hessian_Matvec
            Since PINNs are often small, this is feasible.
            Warning! This operation scales quadratically in time and space!
        '''

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
