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
        return self.embedding_layer(self.str_to_int(inputs))
    
class InputsProcessing(tf.keras.layers.Layer):
    """Do the inputs processing such as normalization or categorical data embedding."""
    
    def __init__(self,
                 categorical_inputs=None,
                 normalize_inputs=False,
                 **kwargs):
        """Initializes the layer.

        Parameters
        ----------
        categorical_inputs : dict
            Keys: The names of categorical inputs; Values: Their corresponding embedding layer (e.g.: StringEmbedding layer).
        normalize_inputs : bool
            Do we do inputs normalization? (default to True)
        """
        super().__init__(**kwargs)
        if categorical_inputs is None:
            self.categorical_inputs = {}
        else:
            self.categorical_inputs = categorical_inputs
        self.normalize_inputs = normalize_inputs
        if self.normalize_inputs:
            self.batch_normalization = tf.keras.layers.BatchNormalization()
        
    def call(self, inputs):
        inputs_tmp = {}
        for one_input_name in inputs.keys():
            if one_input_name in self.categorical_inputs.keys():
                inputs_tmp[one_input_name] = self.categorical_inputs[one_input_name](inputs[one_input_name])
            else:
                inputs_tmp[one_input_name] = inputs[one_input_name]
                        
        embedded_inputs = tf.keras.layers.Concatenate()(list(inputs_tmp.values()))
        
        if self.normalize_inputs:
            embedded_inputs = self.batch_normalization(embedded_inputs)
            
        return embedded_inputs