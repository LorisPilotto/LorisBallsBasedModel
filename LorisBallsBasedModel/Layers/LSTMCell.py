import tensorflow as tf


class LSTMCell(tf.keras.layers.Layer):
    """A LSTM cell."""

    def __init__(self,
                 activation='relu',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 **kwargs
                ):
        """Initilaizes the LSTMCell.
        
        Parameters
        ----------
        TODO"""
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = tf.keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)        
        
    def build(self, input_shape):
        self.units = input_shape[-1]
        self.kernel = self.add_weight(shape=(self.units, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4),
                                                name='recurrent_kernel',
                                                initializer=self.recurrent_initializer,
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint,
                                                dtype=self.dtype,
                                                trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        super().build(input_shape)
        
    def call(self, inputs, states):
        
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        
        z = tf.matmul(inputs, self.kernel)
        z += tf.matmul(h_tm1, self.recurrent_kernel)
        if self.use_bias:
            z += self.bias
        
        z = tf.split(z, num_or_size_splits=4, axis=1)
        
        i = self.recurrent_activation(z[0])
        f = self.recurrent_activation(z[1])
        c = f * c_tm1 + i * self.activation(z[2])
        o = self.recurrent_activation(z[3])
        
        h = o * self.activation(c)
        
        return h, [h, c]