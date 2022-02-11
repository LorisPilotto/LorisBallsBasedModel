import tensorflow as tf
import tensorflow_addons as tfa
from LorisBallsBasedModel.Layers.WeightedAdd import WeightedAdd
from LorisBallsBasedModel.Layers.BoundedParaboloids import BoundedParaboloids
    

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
                 weighted_add_layer=WeightedAdd(),
                 prior_mask_scales_function=None,
                 regularizer=tf.keras.regularizers.L1(0.),
                 activation=tfa.activations.sparsemax,
                 epsilon=1e-8,
                 entropy_weight=0,
                 **kwargs):
        """Initilaizes the AttentiveTransformer.
        
        Parameters
        ----------
        TODO"""
        if input_embedding_layer is not None and (input_dense_units is not None or input_Loris_balls_units is not None):
            raise ValueError("instantiate either `input_embedding_layer` or `input_dense_units`;`input_Loris_balls_units`.")
        if prior_outputs_embedding_layer is not None and (prior_outputs_dense_units is not None or prior_outputs_Loris_balls_units is not None):
            raise ValueError("instantiate either `prior_outputs_embedding_layer` or `prior_outputs_dense_units`;`prior_outputs_Loris_balls_units`.")
        
        super().__init__(**kwargs)
        self.gamma = gamma
        self.dropout_rate = dropout_rate
        self.input_embedding_layer = input_embedding_layer
        if self.input_embedding_layer is None:
            self.input_dense_units = input_dense_units
            self.input_Loris_balls_units = input_Loris_balls_units
        self.weighted_add_layer = weighted_add_layer
        if prior_mask_scales_function is None:
            def prior_mask_scales_function(gamma, prior_masks_list, input_shape):
                gamma = tf.cast(gamma, tf.float32)
                mean_features_importance = tf.reduce_mean(prior_masks_list, 0)
                return tf.pow(gamma, mean_features_importance)
        self.prior_mask_scales_function = prior_mask_scales_function
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.activation = tf.keras.activations.get(activation)
        self.inp_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.inp_emb_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.prior_outputs_embedding_layer = prior_outputs_embedding_layer
        if self.prior_outputs_embedding_layer is None:
            self.prior_outputs_dense_units = prior_outputs_dense_units
            self.prior_outputs_Loris_balls_units = prior_outputs_Loris_balls_units
        self.prior_out_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        
    def build(self, input_shape):
        if self.input_embedding_layer is None:
            if self.input_dense_units is None:
                self.input_dense_units = input_shape[0][1]
            if self.input_Loris_balls_units is None:
                self.input_Loris_balls_units = input_shape[0][1]
            self.input_dense1 = tf.keras.layers.Dense(self.input_dense_units, 'relu')
            self.input_Loris_balls1 = BoundedParaboloids(self.input_Loris_balls_units, processing_layer=tf.keras.layers.BatchNormalization())
            self.input_dense_out = tf.keras.layers.Dense(input_shape[0][1])
        if self.prior_outputs_embedding_layer is not None:
            self.prior_outputs_embedding_layer = self.prior_outputs_embedding_layer(input_shape)
        else:
            if self.prior_outputs_dense_units is None:
                self.prior_outputs_dense_units = input_shape[0][1]
            if self.prior_outputs_Loris_balls_units is None:
                self.prior_outputs_Loris_balls_units = input_shape[0][1]
            self.prior_outputs_dense1 = tf.keras.layers.Dense(self.prior_outputs_dense_units, 'relu')
            self.prior_outputs_Loris_balls1 = BoundedParaboloids(self.prior_outputs_Loris_balls_units, processing_layer=tf.keras.layers.BatchNormalization())
            self.prior_outputs_dense_out = tf.keras.layers.Dense(input_shape[0][1])
        super().build(input_shape)
    
    def call(self, inputs):
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
            prior_outputs_embedding = self.prior_outputs_dense_out(tf.keras.layers.Concatenate()([prior_outputs_dense1, prior_outputs_Loris_balls1]))
        else:
            prior_outputs_embedding = self.prior_outputs_embedding_layer(prior_outputs)
        mask = self.weighted_add_layer([self.inp_drop(input_tensor),
                                        self.inp_emb_drop(inputs_embedding),
                                        self.prior_out_drop(prior_outputs_embedding)])
        prior = self.prior_mask_scales_function(self.gamma, prior_masks_list, tf.shape(input_tensor))
        mask *= prior
        
        self.add_loss(self.regularizer(mask))
        
        mask = self.activation(mask)
        
        entropy_regularization = self.entropy_weight*tf.reduce_mean(
            tf.reduce_sum(-mask * tf.math.log(mask + self.epsilon), axis=-1)
        )
        self.add_loss(entropy_regularization)
        
        return mask
    
class FirstAttentiveTransformer(tf.keras.layers.Layer):
    """The first feature selection layer (do not receive prior information)."""
    
    def __init__(self,
                 dropout_rate=0.,
                 input_dense_units=None,
                 input_Loris_balls_units=None,
                 input_embedding_layer=None,
                 weighted_add_layer=WeightedAdd(),
                 regularizer=tf.keras.regularizers.L1(0.),
                 activation=tfa.activations.sparsemax,
                 epsilon=1e-8,
                 entropy_weight=0,
                 **kwargs):
        """Initilaizes the AttentiveTransformer.
        
        Parameters
        ----------
        TODO"""
        if input_embedding_layer is not None and (input_dense_units is not None or input_Loris_balls_units is not None):
            raise ValueError("instantiate either `input_embedding_layer` or `input_dense_units`;`input_Loris_balls_units`.")
        
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.input_embedding_layer = input_embedding_layer
        if self.input_embedding_layer is None:
            self.input_dense_units = input_dense_units
            self.input_Loris_balls_units = input_Loris_balls_units
        self.weighted_add_layer = weighted_add_layer
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.activation = tf.keras.activations.get(activation)
        self.inp_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.inp_emb_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        
    def build(self, input_shape):
        if self.input_embedding_layer is None:
            if self.input_dense_units is None:
                self.input_dense_units = input_shape[1]
            if self.input_Loris_balls_units is None:
                self.input_Loris_balls_units = input_shape[1]
            self.input_dense1 = tf.keras.layers.Dense(self.input_dense_units, 'relu')
            self.input_Loris_balls1 = BoundedParaboloids(self.input_Loris_balls_units, processing_layer=tf.keras.layers.BatchNormalization())
            self.input_dense_out = tf.keras.layers.Dense(input_shape[1], 'relu')
        super().build(input_shape)
    
    def call(self, inputs):
        if self.input_embedding_layer is None:
            input_dense1 = self.input_dense1(inputs)
            input_Loris_balls1 = self.input_Loris_balls1(inputs)
            inputs_embedding = self.input_dense_out(tf.keras.layers.Concatenate()([input_dense1, input_Loris_balls1]))
        else:
            inputs_embedding = self.input_embedding_layer(inputs)
        
        mask = self.weighted_add_layer([self.inp_drop(inputs),
                                        self.inp_emb_drop(inputs_embedding)])
        
        self.add_loss(self.regularizer(mask))
        
        mask = self.activation(mask)
        
        entropy_regularization = self.entropy_weight*tf.reduce_mean(
            tf.reduce_sum(-mask * tf.math.log(mask + self.epsilon), axis=-1)
        )
        self.add_loss(entropy_regularization)
        
        return mask