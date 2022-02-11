import tensorflow as tf


class StringEmbedding(tf.keras.layers.Layer):
    """An embedding layer used when an input is a categorical string."""
    
    def __init__(self,
                 unique_elements,
                 embedding_size=1,
                 **kwargs):
        """Initializes the layer.
        
        Parameters
        ----------
        unique_elements : list
            A list of all unique elements present in the categorical data.
        embedding_size : int
            The embedding dimension. (default to 1)"""
        super().__init__(**kwargs)
        self.unique_elements = unique_elements
        self.embedding_size = embedding_size
        # needed layers to perform str embedding:
        self.str_to_int = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=self.unique_elements)
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=len(self.unique_elements) + 1,
                                                         output_dim=self.embedding_size)
        
    def call(self, inputs):
        """Transformation from inputs to outputs."""
        converted_to_int = self.str_to_int(inputs)
        converted_to_embedding = self.embedding_layer(converted_to_int)
        return tf.keras.layers.Flatten()(converted_to_embedding)

class UniformNoise(tf.keras.layers.Layer):
    """A layer that add a uniform noise at training time."""
    
    def __init__(self, min_val=-.01, max_val=.01, seed=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.seed = seed
        self.mean = float(self.max_val - self.min_val)/2.

    def call(self, inputs, training=None):
        if training:
            return inputs + tf.random.uniform(tf.shape(inputs),
                                              minval=self.min_val,
                                              maxval=self.max_val,
                                              seed=self.seed,
                                              dtype=inputs.dtype)
        else:
            return inputs + tf.cast(tf.fill(tf.shape(inputs),
                                            self.mean), inputs.dtype)

class InputsProcessing(tf.keras.layers.Layer):
    """Do the inputs processing such as normalization or categorical data embedding."""
    
    def __init__(self,
                 categorical_inputs=None,
                 general_processing_layer=None,
                 **kwargs):
        """Initializes the layer.

        Parameters
        ----------
        categorical_inputs : dict
            Keys: The names of categorical inputs; Values: Their corresponding embedding layer (e.g.: StringEmbedding layer).
        general_processing_layer : tf.keras.layers.Layer
            A layer applied before returning the embedding. (E.g.: tf.keras.layers.BatchNormalization(), tf.keras.layers.GaussianNoise(), UniformNoise())
        """
        super().__init__(**kwargs)
        if categorical_inputs is None:
            self.categorical_inputs = {}
        else:
            self.categorical_inputs = categorical_inputs
        self.general_processing_layer = general_processing_layer
        
    def call(self, inputs):
        inputs_tmp = {}
        for one_input_name in inputs.keys():
            if one_input_name in self.categorical_inputs.keys():
                inputs_tmp[one_input_name] = self.categorical_inputs[one_input_name](inputs[one_input_name])
            else:
                inputs_tmp[one_input_name] = inputs[one_input_name]
                        
        embedded_inputs = tf.keras.layers.Concatenate()(list(inputs_tmp.values()))
        embedded_inputs = tf.keras.layers.Flatten()(embedded_inputs)
        
        if self.general_processing_layer is not None:
            embedded_inputs = self.general_processing_layer(embedded_inputs)
        
        return embedded_inputs