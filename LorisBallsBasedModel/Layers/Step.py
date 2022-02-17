import tensorflow as tf
from LorisBallsBasedModel.Layers.BoundedParaboloids import BoundedParaboloids


class DenseAndBoundedParaboloids(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 dict_dense_in_params=None,
                 dict_bounded_paraboloids_in_params=None,
                 dict_dense_out_params=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        if dict_dense_in_params is None:
            self.dict_dense_in_params = {'units': self.units,
                                         'activation': 'relu'}
        else:
            self.dict_dense_in_params = dict_dense_in_params
            self.dict_dense_in_params['units'] = self.units
        if dict_bounded_paraboloids_in_params is None:
            self.dict_bounded_paraboloids_in_params = {'units': self.units,
                                                       'processing_layer': tf.keras.layers.BatchNormalization()}
        else:
            self.dict_bounded_paraboloids_in_params = dict_bounded_paraboloids_in_params
            self.dict_bounded_paraboloids_in_params['units'] = self.units
            tf.print("BoundedParaboloids inputs normalized?")
        if dict_dense_out_params is None:
            self.dict_dense_out_params = {'units': self.units,
                                          'activation': 'relu'}
        else:
            self.dict_dense_out_params = dict_dense_out_params
            self.dict_dense_out_params['units'] = self.units
        self.dense_in_layer = tf.keras.layers.Dense(**self.dict_dense_in_params)
        self.bounded_paraboloids_in_layer = BoundedParaboloids(**self.dict_bounded_paraboloids_in_params)
        self.dense_out_layer = tf.keras.layers.Dense(**self.dict_dense_out_params)
        
    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        
        i1 = self.dense_in_layer(inputs)
        i2 = self.bounded_paraboloids_in_layer(inputs)
        return self.dense_out_layer(tf.keras.layers.Concatenate()([i1, i2]))
    
class Step(tf.keras.layers.Layer):
    def __init__(self,
                 attentive_transformer,
                 attentive_transformer_params_dict,
                 features_outputs_units,
                 features_pass_next_step_units,
                 features_embedding_layer=None,
                 prior_outputs_embedding_layer=None,
                 prior_outputs_units=None,
                 processing_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.attentive_transformer = attentive_transformer(**attentive_transformer_params_dict)
        self.features_outputs_units = features_outputs_units
        self.features_pass_next_step_units = features_pass_next_step_units
        self.features_embedding_layer = features_embedding_layer
        if self.features_embedding_layer is None:
            self.features_embedding_layer_1 = DenseAndBoundedParaboloids(self.features_outputs_units+self.features_pass_next_step_units)
            self.features_embedding_layer_2 = DenseAndBoundedParaboloids(self.features_outputs_units+self.features_pass_next_step_units)
        self.prior_outputs_embedding_layer = prior_outputs_embedding_layer
        if self.prior_outputs_embedding_layer is None:
            self.prior_outputs_units = prior_outputs_units
            self.prior_outputs_DenseAndBoundedParaboloids_embedding_layer = DenseAndBoundedParaboloids(self.prior_outputs_units)
        self.processing_layer = processing_layer
        
    def call(self, inputs):
        if self.processing_layer is not None:
            inputs = self.processing_layer(inputs)
        
        input_tensor, prior_outputs_list, prior_masks_list = inputs
        
        mask = self.attentive_transformer(inputs)
        
        selected_features = input_tensor*mask
        
        if self.prior_outputs_embedding_layer is None:
            prior_outputs_embedding = self.prior_outputs_DenseAndBoundedParaboloids_embedding_layer(
                tf.keras.layers.Concatenate()(prior_outputs_list)
            )
        else:
            prior_outputs_embedding = self.prior_outputs_embedding_layer(prior_outputs_list)
        
        if self.features_embedding_layer is None:
            inp = tf.keras.layers.Concatenate()([selected_features, prior_outputs_embedding])
            embedding_1 = self.features_embedding_layer_1(inp)
            output = self.features_embedding_layer_2(tf.keras.layers.Concatenate()([embedding_1, inp]))
        else:
            output = self.features_embedding_layer(selected_features, prior_outputs_embedding)
        
        return output[:, :self.features_outputs_units], output[:, self.features_outputs_units:], mask
    
class FirstStep(tf.keras.layers.Layer):
    def __init__(self,
                 attentive_transformer,
                 attentive_transformer_params_dict,
                 features_outputs_units,
                 features_pass_next_step_units,
                 features_embedding_layer=None,
                 processing_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.attentive_transformer = attentive_transformer(**attentive_transformer_params_dict)
        self.features_outputs_units = features_outputs_units
        self.features_pass_next_step_units = features_pass_next_step_units
        self.features_embedding_layer = features_embedding_layer
        if self.features_embedding_layer is None:
            self.features_embedding_layer_1 = DenseAndBoundedParaboloids(self.features_outputs_units+self.features_pass_next_step_units)
            self.features_embedding_layer_2 = DenseAndBoundedParaboloids(self.features_outputs_units+self.features_pass_next_step_units)
        self.processing_layer = processing_layer
        
    def call(self, inputs):
        if self.processing_layer is not None:
            inputs = self.processing_layer(inputs)
        
        mask = self.attentive_transformer(inputs)
        
        selected_features = inputs*mask
        
        if self.features_embedding_layer is None:
            embedding_1 = self.features_embedding_layer_1(selected_features)
            output = self.features_embedding_layer_2(tf.keras.layers.Concatenate()([embedding_1, selected_features]))
        else:
            output = self.features_embedding_layer(selected_features)
        
        return output[:, :self.features_outputs_units], output[:, self.features_outputs_units:], mask