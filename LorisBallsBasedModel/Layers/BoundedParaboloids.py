import tensorflow as tf


class SemiAxisRegularizer(tf.keras.regularizers.Regularizer):
    """A regularizer for the semi axis of BoundedParaboloids layer.
    We do not want excessively small ellipsoidal subspaces that would overfite to the training data."""
    
    def __init__(self, m):
        """Initializes the SemiAxisRegularizer.
        
        Parameters
        ----------
        m : float
            Regularization factor. Increase m to penalize small ellipsoid."""
        self.m = m
        
    def __call__(self, x):
        return self.m*tf.math.reduce_sum(tf.math.divide(1, x))
    
    def get_config(self):
        return {'m': float(self.m)}
    
class MinusOrPlusOnesInitializer(tf.keras.initializers.Initializer):
    """An initializer for the multipliers of BoundedParaboloids layer."""
    
    def __init__(self, p=.5):
        """Initializes the MinusOrPlusOnesInitializer.
        
        Parameters
        ----------
        p : float
            Probability to be 1."""
        self.p = p
        
    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None:
            dtype = tf.keras.backend.floatx()
        dtype = tf.as_dtype(dtype)
        return tf.cast(tf.where(tf.random.uniform(shape, -(1-self.p), self.p, 'float32') > 0, 1., -1.), dtype)

    def get_config(self):    
        return {"p": self.p}
    
class BoundedParaboloids(tf.keras.layers.Layer):
    """With respect to the selected features, highlight ellipsoidal subspaces of the vectorial space."""
    
    def __init__(self,
                 units,
                 semi_axis_initializer=tf.keras.initializers.RandomUniform(minval=.02, maxval=.2),
                 semi_axis_regularizer=SemiAxisRegularizer(0.000001),
                 semi_axis_constraint=lambda x: tf.maximum(x, 10**-5),  # should be strictly positive do avoid division by 0
                 sharpness_initializer=tf.keras.initializers.RandomUniform(minval=-2., maxval=2.),
                 sharpness_regularizer=None,
                 sharpness_constraint=None,
                 shift_initializer=tf.keras.initializers.RandomNormal(mean=.0, stddev=1.),
                 shift_regularizer=None,
                 shift_constraint=None,
                 activation='sigmoid',
                 use_multiplier=True,
                 multiplier_initializer=MinusOrPlusOnesInitializer(),
                 multiplier_regularizer=None,
                 multiplier_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """Initilaizes the BoundedParaboloids.
        
        Parameters
        ----------
        TODO"""
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        self.semi_axis_initializer = tf.keras.initializers.get(semi_axis_initializer)
        self.semi_axis_regularizer = tf.keras.regularizers.get(semi_axis_regularizer)
        self.semi_axis_constraint = tf.keras.constraints.get(semi_axis_constraint)
        self.sharpness_initializer = tf.keras.initializers.get(sharpness_initializer)
        self.sharpness_regularizer = tf.keras.regularizers.get(sharpness_regularizer)
        self.sharpness_constraint = tf.keras.constraints.get(sharpness_constraint)
        self.shift_initializer = tf.keras.initializers.get(shift_initializer)
        self.shift_regularizer = tf.keras.regularizers.get(shift_regularizer)
        self.shift_constraint = tf.keras.constraints.get(shift_constraint)
        self.activation = tf.keras.activations.get(activation)
        self.use_multiplier = use_multiplier
        self.multiplier_initializer = tf.keras.initializers.get(multiplier_initializer)
        self.multiplier_regularizer = tf.keras.regularizers.get(multiplier_regularizer)
        self.multiplier_constraint = tf.keras.constraints.get(multiplier_constraint)
        
    def build(self, input_shape):
        self.semi_axis_list = [self.add_weight('semi_axis_'+str(ellipse_id),
                                               shape=input_shape[1:],
                                               initializer=self.semi_axis_initializer,
                                               regularizer=self.semi_axis_regularizer,
                                               constraint=self.semi_axis_constraint,
                                               dtype=self.dtype,
                                               trainable=True) for ellipse_id in range(self.units)]
        self.sharpness_list = [self.add_weight('sharpness'+str(ellipse_id),
                                               shape=[1],
                                               initializer=self.sharpness_initializer,
                                               regularizer=self.sharpness_regularizer,
                                               constraint=self.sharpness_constraint,
                                               dtype=self.dtype,
                                               trainable=True) for ellipse_id in range(self.units)]
        self.shift_list = [self.add_weight('shift'+str(ellipse_id),
                                           shape=input_shape[1:],
                                           initializer=self.shift_initializer,
                                           regularizer=self.shift_regularizer,
                                           constraint=self.shift_constraint,
                                           dtype=self.dtype,
                                           trainable=True) for ellipse_id in range(self.units)]
        if self.use_multiplier:
            self.multiplier_list = [self.add_weight('multiplier'+str(ellipse_id),
                                                    shape=[1],
                                                    initializer=self.multiplier_initializer,
                                                    regularizer=self.multiplier_regularizer,
                                                    constraint=self.multiplier_constraint,
                                                    dtype=self.dtype,
                                                    trainable=True) for ellipse_id in range(self.units)]
        else:
            self.multiplier_list = None
        super().build(input_shape)
                
    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        
        ellipsoidal_list = []
        
        for ellipse_id in range(self.units):
            shifted_inputs = tf.add(inputs, self.shift_list[ellipse_id])
            
            ellipsoidal = 1 - tf.tensordot(a=tf.square(shifted_inputs), b=1/tf.square(self.semi_axis_list[ellipse_id]), axes=[1, 0])
            ellipsoidal = tf.expand_dims(ellipsoidal, -1)
            
            sharpe_ellipsoidal = self.sharpness_list[ellipse_id]*ellipsoidal
            
            if self.activation is not None:
                sharpe_ellipsoidal = self.activation(sharpe_ellipsoidal)

            if self.use_multiplier:
                sharpe_ellipsoidal = self.multiplier_list[ellipse_id]*sharpe_ellipsoidal
            
            ellipsoidal_list.append(sharpe_ellipsoidal)
            
        return tf.keras.layers.Concatenate()(ellipsoidal_list)