from tensorflow.keras.layers import Dense, Conv2D, Activation, Input, Flatten, AveragePooling2D, MaxPool2D, LeakyReLU, Dropout, Add
from tensorflow.keras.layers import BatchNormalization, Concatenate, Layer, SpatialDropout2D, Conv2DTranspose, UpSampling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class VariationalAutoEncoder(tf.keras.Model):
    """
    Learns to autoencode spatial fields using a variational loss

    Args:
        latent_dim (int): Size of the latent vector
        min_filters (int): Number of convolutional filters in first layer. Grows from there
        filter_growth_rate (float): Factor by how much the number of convolutional filters increases by layer.
        filter_width (int): Width of a single convolutional filter
        min_data_width (int): Minimum width of field after cycling through convolution and pooling layers
        pooling_width (int): Size of pooling window.
        hidden_activation (str): Type of activation function used after conv layers. "leaky" is for LeakyReLU
        pooling (str): either "max" or "mean".
        data_format (str): "channels_last" or "channels_first" depending on whether your data shape is NXYC or NCXY
        leaky_alpha (float): Scaling factor for leaky ReLU
        interpolation (str): For the Upsampling layers, whether to use bilinear or nearest interpolation
        use_l2 (bool): Whether to use l2 regularization or not
        l2_alpha (float): Strength of the l2 regularization if use_l2 is True
        learning_rate (float): Learning rate for Adam optimizer.
        verbose (int): Level of verbosity in fit function.
    """
    def __init__(self, latent_dim=8, min_filters=8,
                 filter_growth_rate=2, filter_width=3, min_data_width=4, pooling_width=2,
                 hidden_activation="relu", pooling="max", data_format="channels_last",
                 leaky_alpha=0.1, interpolation="bilinear", use_l2=False,
                 learning_rate=0.001, l2_alpha=0, verbose=0, **kwargs):
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.min_filters = min_filters
        self.filter_growth_rate = filter_growth_rate
        self.filter_width = filter_width
        self.min_data_width = min_data_width
        self.pooling_width = pooling_width
        self.hidden_activation = hidden_activation
        self.pooling = pooling
        self.data_format = data_format
        self.optimizer = Adam(learning_rate=learning_rate)
        self.leaky_alpha = leaky_alpha
        self.interpolation = interpolation
        self.learning_rate = learning_rate
        self.l2_alpha = l2_alpha
        self.verbose = verbose
        self.use_l2 = use_l2
        self.encoder_net = None
        self.decoder_net = None
        return

    def build_encoder_net(self, conv_input_shape):
        if self.use_l2:
            reg = l2(self.l2_alpha)
        else:
            reg = None
        conv_input_layer = Input(shape=conv_input_shape, name="conv_input")
        num_conv_layers = int(np.round((np.log(conv_input_shape[1]) - np.log(self.min_data_width))
                                       / np.log(self.pooling_width)))
        num_filters = self.min_filters
        scn_model = conv_input_layer
        for c in range(num_conv_layers):
            scn_model = Conv2D(num_filters, (self.filter_width, self.filter_width),
                               data_format=self.data_format,
                               kernel_regularizer=reg, padding="same", name="conv_{0:02d}".format(c))(scn_model)
            if self.hidden_activation == "leaky":
                scn_model = LeakyReLU(self.leaky_alpha, name="hidden_activation_{0:02d}".format(c))(scn_model)
            else:
                scn_model = Activation(self.hidden_activation, name="hidden_activation_{0:02d}".format(c))(scn_model)
            num_filters = int(num_filters * self.filter_growth_rate)
            if self.pooling.lower() == "max":
                scn_model = MaxPool2D(pool_size=(self.pooling_width, self.pooling_width),
                                      data_format=self.data_format, name="pooling_{0:02d}".format(c))(scn_model)
            else:
                scn_model = AveragePooling2D(pool_size=(self.pooling_width, self.pooling_width),
                                             data_format=self.data_format, name="pooling_{0:02d}".format(c))(scn_model)
        scn_model = Conv2D(num_filters, (self.filter_width, self.filter_width),
                           data_format=self.data_format,
                           kernel_regularizer=reg, padding="same", name="conv_{0:02d}".format(num_conv_layers))(scn_model)
        if self.hidden_activation == "leaky":
            scn_model = LeakyReLU(self.leaky_alpha, name="hidden_activation_{0:02d}".format(num_conv_layers))(scn_model)
        else:
            scn_model = Activation(self.hidden_activation, name="hidden_activation_{0:02d}".format(num_conv_layers))(scn_model)
        scn_model = Flatten(name="flatten")(scn_model)
        scn_model = Dense(self.latent_dim * 2, name="dense_output")(scn_model)
        full_model = Model(conv_input_layer, scn_model)
        return full_model

    def build_decoder_net(self, conv_input_shape):
        decoder_input_layer = Input(shape=self.latent_dim, name="dec_input")
        num_conv_layers = int(np.round((np.log(conv_input_shape[1]) - np.log(self.min_data_width))
                                       / np.log(self.pooling_width)))
        num_filters = int(self.min_filters * self.filter_growth_rate ** num_conv_layers)
        dec_model = Dense(num_filters * self.min_data_width * self.min_data_width)(decoder_input_layer)
        if self.hidden_activation == "leaky":
            dec_model = LeakyReLU(self.leaky_alpha, name="hidden_activation_{0:02d}".format(-1))(dec_model)
        else:
            dec_model = Activation(self.hidden_activation, name="hidden_activation_{0:02d}".format(-1))(dec_model)
        dec_model = Reshape(target_shape=(self.min_data_width, self.min_data_width, num_filters))(dec_model)
        if self.use_l2:
            reg = l2(self.l2_alpha)
        else:
            reg = None
        for c in range(num_conv_layers):
            num_filters = int(num_filters / self.filter_growth_rate)
            dec_model = Conv2DTranspose(num_filters,
                                        kernel_size=(self.filter_width, self.filter_width), padding="same",
                                        data_format=self.data_format, kernel_regularizer=reg)(dec_model)
            if self.hidden_activation == "leaky":
                dec_model = LeakyReLU(self.leaky_alpha, name="hidden_activation_{0:02d}".format(c))(dec_model)
            else:
                dec_model = Activation(self.hidden_activation, name="hidden_activation_{0:02d}".format(c))(dec_model)
            dec_model = UpSampling2D(size=(self.pooling_width, self.pooling_width),
                                     interpolation=self.interpolation)(dec_model)
        dec_model = Conv2D(conv_input_shape[-1], (self.filter_width, self.filter_width), padding="same")(dec_model)
        return Model(decoder_input_layer, dec_model)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    @staticmethod
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)
    
    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        squared_error = (x_logit - x) ** 2
        mse = tf.reduce_mean(squared_error, axis=[1, 2, 3])
        #cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        #logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(-mse + logpz - logqz_x)

    @tf.function
    def compute_apply_gradients(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def fit(self, x=None, y=None, epochs=1, batch_size=64, shuffle=True, **kwargs):
        x_shape = x.shape[1:]
        self.encoder_net = self.build_encoder_net(x_shape)
        self.decoder_net = self.build_decoder_net(x_shape)
        if self.verbose > 0:
            print(self.encoder_net.summary())
            print(self.decoder_net.summary())
        for epoch in range(1, epochs + 1):
            if self.verbose > 0:
                print("Epoch:", epoch)
            indices = np.arange(x.shape[0])
            if shuffle:
                indices = np.random.permutation(indices)
            batch_indices = np.append(np.arange(0, indices.size, batch_size), indices.size)
            for b in range(batch_indices.size - 1):
                x_batch = x[indices[batch_indices[b]:batch_indices[b+1]]]
                self.compute_apply_gradients(x_batch, self.optimizer)
                if self.verbose > 1:
                    print(f"Batch {b:03d} Loss", self.compute_loss(x_batch).numpy())
        return
