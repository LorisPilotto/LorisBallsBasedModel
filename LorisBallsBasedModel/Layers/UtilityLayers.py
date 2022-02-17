import tensorflow as tf


class IdentityLayer(tf.keras.layers.Layer):
    """A layer that pass the input. Used to apply an activity_regularizer (well divided by the batch size)."""
    
    def __init__(self,
                 activity_regularizer,
                 **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.activity_regularizer = activity_regularizer
        
    def call(self, inputs):
        return inputs