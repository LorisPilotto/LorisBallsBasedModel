import tensorflow as tf
import tensorflow_addons as tfa
from LorisBallsBasedModel.Layers.WeightedAdd import WeightedAdd
from LorisBallsBasedModel.Layers.BoundedParaboloids import BoundedParaboloids


class TensorRegularizer(tf.keras.layers.Layer):
    """A layer that add an activity penalty on a tensor."""
    def __init__(self,
                 activity_regularizer,
                 **kwargs):
        """Initilaizes the TensorRegularizer.
        
        Parameters
        ----------
        TODO"""
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        
    def call(self, inputs):
        return inputs
    

class AttentiveTransformer(tf.keras.layers.Layer):
    """The feature selection layer."""
    
    def __init__(self,
                 gamma,
                 dropout_rate=0.,
                 input_dense_units=None,
                 input_Loris_balls_units=None,
                 input_embedding_layer=None,
                 prior_outputs_list=None,
                 prior_outputs_dense_units=None,
                 prior_outputs_Loris_balls_units=None,
                 prior_outputs_embedding_layer=None,
                 weighted_add_layer=WeightedAdd(use_bias=True),
                 prior_masks_list=None,
                 prior_function=None,
                 regularizers=tf.keras.regularizers.L1(0.),
                 activation=tfa.layers.Sparsemax(),
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
        self.prior_masks_list = prior_masks_list
        if prior_function is None:
            if self.prior_masks_list is None:
                def prior_function(gamma, prior_masks_list, input_shape):
                    return tf.constant(1, shape=input_shape)
            else:
                def prior_function(gamma, prior_masks_list, input_shape):
                    mean_features_importance = tf.reduce_mean(prior_masks_list, 0)
                    return tf.pow(gamma, mean_features_importance)
        self.prior_function = prior_function
        self.tensor_regularizer = TensorRegularizer(regularizers)
        self.activation = activation
        self.inp_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.inp_emb_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.prior_outputs_list = prior_outputs_list
        if self.prior_outputs_list is not None:
            self.prior_outputs_embedding_layer = prior_outputs_embedding_layer
            if self.prior_outputs_embedding_layer is None:
                self.prior_outputs_dense_units = prior_outputs_dense_units
                self.prior_outputs_Loris_balls_units = prior_outputs_Loris_balls_units
            self.prior_out_drop = tf.keras.layers.Dropout(self.dropout_rate)
        
    def build(self, input_shape):
        if self.input_embedding_layer is None:
            if self.input_dense_units is None:
                self.input_dense_units = input_shape[1]
            if self.input_Loris_balls_units is None:
                self.input_Loris_balls_units = input_shape[1]
            self.input_dense1 = tf.keras.dense(self.input_dense_units, 'relu')
            self.input_Loris_balls1 = BoundedParaboloids(self.input_Loris_balls_units)
            self.input_dense_out = tf.keras.dense(input_shape[1], 'relu')
        if self.prior_outputs_list is not None:
            if self.prior_outputs_embedding_layer is None:
                if self.prior_outputs_dense_units is None:
                    self.prior_outputs_dense_units = input_shape[1]
                if self.prior_outputs_Loris_balls_units is None:
                    self.prior_outputs_Loris_balls_units = input_shape[1]
                self.prior_outputs_dense1 = tf.keras.dense(self.prior_outputs_dense_units, 'relu')
                self.prior_outputs_Loris_balls1 = BoundedParaboloids(self.prior_outputs_Loris_balls_units)
                self.prior_outputs_dense_out = tf.keras.dense(input_shape[1], 'relu')
    
    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        
        if self.input_embedding_layer is None:
            input_dense1 = self.input_dense1(inputs)
            input_Loris_balls1 = self.input_Loris_balls1(inputs)
            inputs_embedding = self.input_dense_out(tf.keras.layers.Concatenate()([input_dense1, input_Loris_balls1]))
        else:
            inputs_embedding = self.input_embedding_layer(inputs)
        if self.prior_outputs_list is not None:
            prior_outputs = tf.keras.layers.Concatenate()(self.prior_outputs_list)
            if self.prior_outputs_embedding_layer is None:
                prior_outputs_dense1 = self.prior_outputs_dense1(prior_outputs)
                prior_outputs_Loris_balls1 = self.prior_outputs_Loris_balls1(prior_outputs)
                prior_outputs_embedding = self.prior_outputs_dense_out(tf.keras.layers.Concatenate()([prior_outputs_dense1, prior_outputs_Loris_balls1]))
            else:
                prior_outputs_embedding = self.prior_outputs_embedding_layer(prior_outputs)
            mask = self.weighted_add_layer([self.inp_drop(inputs),
                                            self.inp_emb_drop(inputs_embedding),
                                            self.prior_out_drop(prior_outputs_embedding)])
        else:
            mask = self.weighted_add_layer([self.inp_drop(inputs),
                                            self.inp_emb_drop(inputs_embedding)])
        prior = self.prior_function(self.gamma, self.prior_masks_list, input_shape)
        mask *= prior
        
        return self.activation(self.tensor_regularizer(mask))