import tensorflow as tf


class ActivationAsRegularization(tf.keras.regularizers.Regularizer):
    """A regularizer applied before the mask activation.
    We want a penalization that is asymmetric (unlike L1 and L2 Reg.) to force useless features to have negative values.
    For that we penalize high values (and not low values) using some famous functions: ReLU, sigmoid ...
    WARNING: Some activation as regularization (like SELU) can have a negative penalization. Values could shift to -inf."""
    
    def __init__(self,
                 m,
                 activation_as_regularization_function='relu'):
        """Initializes the ActivationAsRegularization.
        
        Parameters
        ----------
        m : float
            Regularization factor. Increase m to increase penalization.
        activation_as_regularization_function : tf.keras.layers.Activation
            The activation function used as penalization. (default to 'relu')
        """
        self.m = m
        self.activation_as_regularization_function = tf.keras.activations.get(activation_as_regularization_function)
        
    def __call__(self, x):
        return self.m*tf.math.reduce_sum(self.activation_as_regularization_function(x))
    
class ShiftedActivation(tf.keras.layers.Layer):
    """Learn an optimal shifting of the activation function."""
    
    def __init__(self,
                 activation,
                 shift_initializer=tf.keras.initializers.Zeros(),
                 shift_regularizer=None,
                 shift_constraint=None,
                 scale_activation_function=False,
                 scale_initializer=tf.keras.initializers.Ones(),
                 scale_regularizer=None,
                 scale_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.shift_initializer = tf.keras.initializers.get(shift_initializer)
        self.shift_regularizer = tf.keras.regularizers.get(shift_regularizer)
        self.shift_constraint = tf.keras.constraints.get(shift_constraint)
        self.scale_activation_function = scale_activation_function
        self.scale_initializer = tf.keras.initializers.get(scale_initializer)
        self.scale_regularizer = tf.keras.regularizers.get(scale_regularizer)
        self.scale_constraint = tf.keras.constraints.get(scale_constraint)
        
    def build(self, input_shape):
        self.shift = self.add_weight('shift',
                                     shape=input_shape[1:],
                                     initializer=self.shift_initializer,
                                     regularizer=self.shift_regularizer,
                                     constraint=self.shift_constraint,
                                     dtype=self.dtype,
                                     trainable=True)
        if self.scale_activation_function:
            self.scale = self.add_weight('scale',
                                         shape=input_shape[1:],
                                         initializer=self.scale_initializer,
                                         regularizer=self.scale_regularizer,
                                         constraint=self.scale_constraint,
                                         dtype=self.dtype,
                                         trainable=True)
        super().build(input_shape)
        
    def call(self, inputs):
        inputs += self.shift
        if self.scale_activation_function:
            inputs *= self.scale
        return self.activation(inputs)