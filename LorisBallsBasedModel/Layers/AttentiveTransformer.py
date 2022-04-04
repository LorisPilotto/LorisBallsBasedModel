import tensorflow as tf
import tensorflow_addons as tfa
from LorisBallsBasedModel.Layers.WeightedAdd import WeightedAdd
from LorisBallsBasedModel.Layers.BoundedParaboloids import BoundedParaboloids
from LorisBallsBasedModel.Layers.UtilityLayers import IdentityLayer


class EntropyRegularization(tf.keras.regularizers.Regularizer):
    """Apply an entropy regularization."""
    
    def __init__(self,
                 m,
                 epsilon=1e-8):
        """Initializes the EntropyRegularization.
        
        Parameters
        ----------
        m : float
            Regularization factor. Increase m to increase penalization.
        epsilon : float
            A small term for stability: log(0+epsilon). (default to 1e-8)
        """
        self.m = m
        self.epsilon = epsilon
        
    def __call__(self, x):
        return self.m*tf.reduce_mean(
            tf.reduce_sum(-x * tf.math.log(x + self.epsilon), axis=-1)
        )

class AttentiveTransformer(tf.keras.layers.Layer):
    """The feature selection layer."""
    
    def __init__(self,
                 gamma,
                 dropout_rate=0.,
                 input_dense_units=None,
                 input_Loris_balls_units=None,
                 input_embedding_layer=None,
                 prior_outputs_dense_units=None,
                 prior_outputs_Loris_balls_units=None,
                 prior_outputs_embedding_layer=None,
                 memory_cell=None,
                 weighted_add_layer=WeightedAdd(use_bias=True),
                 prior_mask_scales_function=None,
                 regularizer=tf.keras.regularizers.L1(0.),
                 activation=tfa.activations.sparsemax,
                 entropy_weight=0.,
                 activity_regularizer=None,
                 **kwargs):
        """Initilaizes the AttentiveTransformer.
        
        Parameters
        ----------
        TODO"""
        if input_embedding_layer is not None and (input_dense_units is not None or input_Loris_balls_units is not None):
            raise ValueError("instantiate either `input_embedding_layer` or `input_dense_units`;`input_Loris_balls_units`.")
        if prior_outputs_embedding_layer is not None and (prior_outputs_dense_units is not None or prior_outputs_Loris_balls_units is not None):
            raise ValueError("instantiate either `prior_outputs_embedding_layer` or `prior_outputs_dense_units`;`prior_outputs_Loris_balls_units`.")
        
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.gamma = gamma
        self.dropout_rate = dropout_rate
        self.input_embedding_layer = input_embedding_layer
        if self.input_embedding_layer is None:
            self.input_dense_units = input_dense_units
            self.input_Loris_balls_units = input_Loris_balls_units
        self.weighted_add_layer = weighted_add_layer
        self.memory_cell = memory_cell
        if prior_mask_scales_function is None:
            def prior_mask_scales_function(gamma, prior_masks_list, input_shape):
                gamma = tf.cast(gamma, tf.float32)
                mean_features_importance = tf.reduce_mean(prior_masks_list, 0)
                return tf.pow(gamma, mean_features_importance)
        self.prior_mask_scales_function = prior_mask_scales_function
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.activity_regularizer_layer = IdentityLayer(self.regularizer)
        self.activation = tf.keras.activations.get(activation)
        self.inp_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.inp_emb_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.inp_RNN_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.prior_outputs_embedding_layer = prior_outputs_embedding_layer
        if self.prior_outputs_embedding_layer is None:
            self.prior_outputs_dense_units = prior_outputs_dense_units
            self.prior_outputs_Loris_balls_units = prior_outputs_Loris_balls_units
        self.prior_out_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.entropy_weight = entropy_weight
        self.activity_entropy_regularizer_layer = IdentityLayer(EntropyRegularization(self.entropy_weight))
        
    def build(self, input_shape):
        if self.input_embedding_layer is None:
            if self.input_dense_units is None:
                self.input_dense_units = input_shape[0][1]
            if self.input_Loris_balls_units is None:
                self.input_Loris_balls_units = input_shape[0][1]
            self.input_dense1 = tf.keras.layers.Dense(self.input_dense_units, 'relu')
            self.input_Loris_balls1 = BoundedParaboloids(self.input_Loris_balls_units,
                                                         processing_layer=tf.keras.layers.BatchNormalization())
            self.input_dense_out = tf.keras.layers.Dense(input_shape[0][1])
        if self.prior_outputs_embedding_layer is not None:
            self.prior_outputs_embedding_layer = self.prior_outputs_embedding_layer(input_shape[0, 1])
        else:
            if self.prior_outputs_dense_units is None:
                self.prior_outputs_dense_units = input_shape[0][1]
            if self.prior_outputs_Loris_balls_units is None:
                self.prior_outputs_Loris_balls_units = input_shape[0][1]
            self.prior_outputs_dense1 = tf.keras.layers.Dense(self.prior_outputs_dense_units, 'relu')
            self.prior_outputs_Loris_balls1 = BoundedParaboloids(self.prior_outputs_Loris_balls_units,
                                                                 processing_layer=tf.keras.layers.BatchNormalization())
            self.prior_outputs_dense_out = tf.keras.layers.Dense(input_shape[0][1])
        super().build(input_shape)
    
    def call(self, inputs, states=None):
        input_tensor, prior_outputs_list, prior_masks_list = inputs
        
        if self.input_embedding_layer is None:
            input_dense1 = self.input_dense1(input_tensor)
            input_Loris_balls1 = self.input_Loris_balls1(input_tensor)
            inputs_embedding = self.input_dense_out(tf.keras.layers.Concatenate()([input_dense1, input_Loris_balls1]))
        else:
            inputs_embedding = self.input_embedding_layer(input_tensor)
        
        prior_outputs = tf.keras.layers.Concatenate()(prior_outputs_list)
        if self.prior_outputs_embedding_layer is None:
            prior_outputs_dense1 = self.prior_outputs_dense1(prior_outputs)
            prior_outputs_Loris_balls1 = self.prior_outputs_Loris_balls1(prior_outputs)
            prior_outputs_embedding = self.prior_outputs_dense_out(tf.keras.layers.Concatenate()([prior_outputs_dense1,
                                                                                                  prior_outputs_Loris_balls1]))
        else:
            prior_outputs_embedding = self.prior_outputs_embedding_layer(prior_outputs)
        
        inputs_RNN, states = self.memory_cell(input_tensor, states)
        
        mask = self.weighted_add_layer([self.inp_drop(input_tensor),
                                        self.inp_emb_drop(inputs_embedding),
                                        self.prior_out_drop(prior_outputs_embedding),
                                        self.inp_RNN_drop(inputs_RNN)])
        
        prior = self.prior_mask_scales_function(self.gamma, prior_masks_list, tf.shape(input_tensor))
        mask *= prior
        
        mask = self.activity_regularizer_layer(mask)
        
        mask = self.activation(mask)
        
        mask = self.activity_entropy_regularizer_layer(mask)
        
        return mask, states
    
class FirstAttentiveTransformer(tf.keras.layers.Layer):
    """The first feature selection layer (do not receive prior information)."""
    
    def __init__(self,
                 dropout_rate=0.,
                 input_dense_units=None,
                 input_Loris_balls_units=None,
                 input_embedding_layer=None,
                 memory_cell=None,
                 weighted_add_layer=WeightedAdd(use_bias=True),
                 regularizer=tf.keras.regularizers.L1(0.),
                 activation=tfa.activations.sparsemax,
                 entropy_weight=0.,
                 activity_regularizer=None,
                 **kwargs):
        """Initilaizes the AttentiveTransformer.
        
        Parameters
        ----------
        TODO"""
        if input_embedding_layer is not None and (input_dense_units is not None or input_Loris_balls_units is not None):
            raise ValueError("instantiate either `input_embedding_layer` or `input_dense_units`;`input_Loris_balls_units`.")
        
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.dropout_rate = dropout_rate
        self.input_embedding_layer = input_embedding_layer
        if self.input_embedding_layer is None:
            self.input_dense_units = input_dense_units
            self.input_Loris_balls_units = input_Loris_balls_units
        self.weighted_add_layer = weighted_add_layer
        self.memory_cell = memory_cell
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.activity_regularizer_layer = IdentityLayer(self.regularizer)
        self.activation = tf.keras.activations.get(activation)
        self.inp_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.inp_emb_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.inp_RNN_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.entropy_weight = entropy_weight
        self.activity_entropy_regularizer_layer = IdentityLayer(EntropyRegularization(self.entropy_weight))
        
    def build(self, input_shape):
        if self.input_embedding_layer is None:
            if self.input_dense_units is None:
                self.input_dense_units = input_shape[1]
            if self.input_Loris_balls_units is None:
                self.input_Loris_balls_units = input_shape[1]
            self.input_dense1 = tf.keras.layers.Dense(self.input_dense_units, 'relu')
            self.input_Loris_balls1 = BoundedParaboloids(self.input_Loris_balls_units,
                                                         processing_layer=tf.keras.layers.BatchNormalization())
            self.input_dense_out = tf.keras.layers.Dense(input_shape[1])
        super().build(input_shape)
    
    def call(self, inputs):
        if self.input_embedding_layer is None:
            input_dense1 = self.input_dense1(inputs)
            input_Loris_balls1 = self.input_Loris_balls1(inputs)
            inputs_embedding = self.input_dense_out(tf.keras.layers.Concatenate()([input_dense1, input_Loris_balls1]))
        else:
            inputs_embedding = self.input_embedding_layer(inputs)
        
        states = [tf.zeros(tf.shape(inputs)), tf.zeros(tf.shape(inputs))]
        inputs_RNN, states = self.memory_cell(inputs, states)
        
        mask = self.weighted_add_layer([self.inp_drop(inputs),
                                        self.inp_emb_drop(inputs_embedding),
                                        self.inp_RNN_drop(inputs_RNN)])
        
        mask = self.activity_regularizer_layer(mask)
        
        mask = self.activation(mask)
        
        mask = self.activity_entropy_regularizer_layer(mask)
        
        return mask, states