import tensorflow as tf
from LorisBallsBasedModel.Layers.Step import Step, FirstStep
from LorisBallsBasedModel.Layers.Processing import InputsProcessing


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
        
    
"""class LorisBallsBasedModel(tf.keras.Model):
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
            self.base_model.unfreeze_model()"""

class LorisBallsBasedModel(tf.keras.Model):
    def __init__(self,
                 output_layer,
                 nbr_steps,
                 first_step_args,
                 step_args=None,
                 first_step_layer=FirstStep,
                 step_layer=Step,
                 input_processing_layer=None,
                 **kwargs):
        if nbr_steps < 1:
            raise ValueError("Give a 'nbr_steps' strictly higher than 0.")
        if nbr_steps > 1 and step_args is None:
            raise ValueError("Give a 'step_args' (list or dict) for steps 2 and higher.")
        
        super().__init__(**kwargs)
        self.nbr_steps = nbr_steps
        self.first_step_layer = first_step_layer
        self.first_step_args = first_step_args
        self.first_step_layer = self.first_step_layer(**self.first_step_args)
        self.step_args = step_args
        if self.nbr_steps > 1:
            self.step_layer = step_layer
            if isinstance(self.step_args, list):
                if len(self.step_args) != self.nbr_steps-1:
                    raise ValueError(f"'step_args' should be of size {self.nbr_steps-1} (i.e. nbr_steps-1).")
                self.steps_list = [self.step_layer(**args) for args in self.step_args]
            else:
                self.steps_list = [self.step_layer(**self.step_args) for s in range(1, self.nbr_steps)]
        self.input_processing_layer = input_processing_layer
        self.output_layer = output_layer
        
    def forward(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        
        if self.input_processing_layer is not None:
            inputs = self.input_processing_layer(inputs)
            
        output, first_step_embedding, first_mask = self.first_step_layer(inputs)
        prior_outputs_list = [first_step_embedding]
        prior_masks_list = [first_mask]
        
        for s in range(0, self.nbr_steps-1):
            tmp_output, tmp_step_embedding, tmp_mask = self.steps_list[s]([inputs,
                                                                           prior_outputs_list,
                                                                           prior_masks_list])
            output += tmp_output
            prior_outputs_list.append(tmp_step_embedding)
            prior_masks_list.append(tmp_mask)
        #tf.print(prior_masks_list)
        
        return self.output_layer(output), prior_masks_list
    
    def call(self, inputs):
        """if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        
        if self.input_processing_layer is not None:
            inputs = self.input_processing_layer(inputs)
            
        output, first_step_embedding, first_mask = self.first_step_layer(inputs)
        prior_outputs_list = [first_step_embedding]
        prior_masks_list = [first_mask]
        
        for s in range(0, self.nbr_steps-1):
            tmp_output, tmp_step_embedding, tmp_mask = self.steps_list[s]([inputs,
                                                                           prior_outputs_list,
                                                                           prior_masks_list])
            output += tmp_output
            prior_outputs_list.append(tmp_step_embedding)
            prior_masks_list.append(tmp_mask)
        #tf.print(prior_masks_list)
        
        return self.output_layer(output)"""
        return self.forward(inputs)[0]
    
    def masks_explain(self, inputs):
        return self.forward(inputs)[1]
    
class LorisBallsBasedModelTransferLearning(tf.keras.Model):
    def __init__(self,
                 step_layer_list,
                 output_layer,
                 input_processing_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.step_layer_list = step_layer_list
        self.output_layer = output_layer
        self.input_processing_layer = input_processing_layer
        
    def call(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        
        if self.input_processing_layer is not None:
            inputs = self.input_processing_layer(inputs)
            
        output, first_step_embedding, first_mask = self.step_layer_list[0](inputs)
        prior_outputs_list = [first_step_embedding]
        prior_masks_list = [first_mask]
        
        for step in self.step_layer_list[1:]:
            tmp_output, tmp_step_embedding, tmp_mask = step([inputs,
                                                             prior_outputs_list,
                                                             prior_masks_list])
            output += tmp_output
            prior_outputs_list.append(tmp_step_embedding)
            prior_masks_list.append(tmp_mask)
                
        return self.output_layer(output)