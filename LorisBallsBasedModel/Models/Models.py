import tensorflow as tf


class SingleLayerPerceptron(tf.keras.Model):
    def __init__(self,
                 output_layer,
                 processing_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.output_layer = output_layer
        self.processing_layer = processing_layer
        
    def call(self, inputs):
        if self.processing_layer is not None:
            inputs = self.processing_layer(inputs)
        return self.output_layer(inputs)

    def freeze_model(self):
        self.trainable = False
        
    def unfreeze_model(self):
        self.trainable = True
        
        
class MultiLayerPerceptron(tf.keras.Model):
    def __init__(self,
                 layers_list,
                 processing_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.layers_list = layers_list
        self.processing_layer = processing_layer
        
    def call(self, inputs):
        if self.processing_layer is not None:
            inputs = self.processing_layer(inputs)
        for a_layer in self.layers_list:
            inputs = a_layer(inputs)
        return inputs
        
    def freeze_model(self):
        self.trainable = False
        
    def unfreeze_model(self):
        self.trainable = True
        
    
class LorisBallsBasedModel(tf.keras.Model):
    def __init__(self,
                 bounded_paraboloids_layer,
                 output_layer,
                 processing_layer=None,
                 processing_layer_used_for_base_model_too=False,  # base model probably already has its own processing layer
                 base_model=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.bounded_paraboloids_layer = bounded_paraboloids_layer
        self.output_layer = output_layer
        self.processing_layer = processing_layer
        self.processing_layer_used_for_base_model_too = processing_layer_used_for_base_model_too
        self.base_model = base_model
    
    def call(self, inputs):
        if self.processing_layer is not None:
            bounded_paraboloids_inputs = self.processing_layer(inputs)
        else:
            bounded_paraboloids_inputs = inputs
        bounded_paraboloids = self.bounded_paraboloids_layer(bounded_paraboloids_inputs)
        if self.base_model is None:
            return self.output_layer(bounded_paraboloids)
        else:
            if self.processing_layer_used_for_base_model_too:
                base_model_inputs = bounded_paraboloids_inputs
            else:
                base_model_inputs = inputs
            base_prediction = self.base_model(base_model_inputs, training=False)  # keep base_model in inference mode (in case of BatchNormalization)
            return self.output_layer(tf.keras.layers.Concatenate()([bounded_paraboloids, base_prediction]))
    
    def freeze_model(self):
        self.trainable = False
    
    def unfreeze_model(self):
        self.trainable = True
    
    def freeze_base_model(self):
        if self.base_model is not None:
            self.base_model.freeze_model()
    
    def unfreeze_base_model(self):
        if self.base_model is not None:
            self.base_model.unfreeze_model()