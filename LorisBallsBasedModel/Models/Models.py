import tensorflow as tf
from LorisBallsBasedModel.Layers.Step import Step, FirstStep
from LorisBallsBasedModel.Layers.Processing import InputsProcessing
from LorisBallsBasedModel.Layers.LSTMCell import LSTMCell


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
    
class LorisBallsBasedModel(tf.keras.Model):
    def __init__(self,
                 output_layer,
                 nbr_steps,
                 first_step_args,
                 first_step_layer=FirstStep,
                 step_args=None,
                 step_layer=Step,
                 input_processing_layer=None,
                 **kwargs):
        if nbr_steps < 1:
            raise ValueError("Give a 'nbr_steps' strictly higher than 0.")
        if nbr_steps > 1 and step_args is None:
            raise ValueError("Give a 'step_args' (list or dict) for steps 2 and higher.")
        
        super().__init__(**kwargs)
        self.input_processing_layer = input_processing_layer
        self.nbr_steps = nbr_steps
        self.first_step_args = first_step_args
        memory_cell = LSTMCell()
        if 'memory_cell' not in self.first_step_args['attentive_transformer_params_dict'].keys():
            self.first_step_args['attentive_transformer_params_dict']['memory_cell'] = memory_cell
        self.first_step_layer = first_step_layer(**self.first_step_args)
        if self.nbr_steps > 1:
            self.step_args = step_args
            if isinstance(self.step_args, list):
                if len(self.step_args) != self.nbr_steps-1:
                    raise ValueError(f"'step_args' should be of size {self.nbr_steps-1} (i.e. nbr_steps-1).")
                self.steps_list = []
                for args in self.step_args:
                    if 'memory_cell' not in args['attentive_transformer_params_dict'].keys():
                        args['attentive_transformer_params_dict']['memory_cell'] = memory_cell
                    self.steps_list.append(step_layer(**args))
            else:
                if 'memory_cell' not in self.step_args['attentive_transformer_params_dict'].keys():
                    self.step_args['attentive_transformer_params_dict']['memory_cell'] = memory_cell
                self.steps_list = [step_layer(**self.step_args) for s in range(self.nbr_steps-1)]
        self.output_layer = output_layer
        
    def forward(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        
        if self.input_processing_layer is not None:
            inputs = self.input_processing_layer(inputs)
            
        [embedding, embedding_pass_next_step, first_mask], states = self.first_step_layer(inputs)
        prior_embeddings_list = [embedding_pass_next_step]
        prior_masks_list = [first_mask]
        
        if self.nbr_steps > 1:
            for step in self.steps_list:
                [tmp_embedding, tmp_embedding_pass_next_step, tmp_mask], states = step([inputs,
                                                                                        prior_embeddings_list,
                                                                                        prior_masks_list],
                                                                                       states)
                embedding += tmp_embedding
                prior_embeddings_list.append(tmp_embedding_pass_next_step)
                prior_masks_list.append(tmp_mask)
        
        if self.output_layer is None:  # For stacked model
            output = embedding
        else:
            output = self.output_layer(embedding)
        
        return output, prior_masks_list, states
    
    def call(self, inputs):
        return self.forward(inputs)[0]
    
    def masks_explain(self, inputs):
        return self.forward(inputs)[1]
    
    def final_states(self, inputs):
        return self.forward(inputs)[2]
    
class LorisBallsBasedModelTransferLearning(tf.keras.Model):
    def __init__(self,
                 step_layers_list,
                 output_layer,
                 input_processing_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.step_layers_list = step_layers_list
        self.output_layer = output_layer
        self.input_processing_layer = input_processing_layer
        
    def forward(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        
        if self.input_processing_layer is not None:
            inputs = self.input_processing_layer(inputs)
            
        [embedding, embedding_pass_next_step, first_mask], states = self.step_layers_list[0](inputs)
        prior_embeddings_list = [embedding_pass_next_step]
        prior_masks_list = [first_mask]
        
        for step in self.step_layers_list[1:]:
            [tmp_embedding, tmp_embedding_pass_next_step, tmp_mask], states = step([inputs,
                                                                                    prior_embeddings_list,
                                                                                    prior_masks_list],
                                                                                   states)
            embedding += tmp_embedding
            prior_embeddings_list.append(tmp_embedding_pass_next_step)
            prior_masks_list.append(tmp_mask)
        
        if self.output_layer is None:  # For stacked model
            output = embedding
        else:
            output = self.output_layer(embedding)
        
        return output, prior_masks_list, states
    
    def call(self, inputs):
        return self.forward(inputs)[0]
    
    def masks_explain(self, inputs):
        return self.forward(inputs)[1]
    
    def final_states(self, inputs):
        return self.forward(inputs)[2]
    
class StackedLorisBallsBasedModels(tf.keras.Model):
    def __init__(self,
                 output_layer,
                 nbr_models,
                 models_args,
                 models=LorisBallsBasedModel,
                 input_processing_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_processing_layer = input_processing_layer
        self.nbr_models = nbr_models
        self.models_args = models_args
        if isinstance(self.models_args, list):
            if len(self.models_args) != self.nbr_models:
                raise ValueError(f"'models_args' should be of size {self.nbr_models} (i.e. nbr_models).")
            if isinstance(models, list):
                if len(models) != len(self.models_args):
                    raise ValueError(f"'models' should be of size {self.models_args} (i.e. models_args).")
                self.models_list = [models[args_id](**args) for args_id, args in enumerate(self.models_args)]
            else:
                self.models_list = [models(**args) for args in self.models_args]
        else:
            self.models_list = [models(**self.models_args) for s in range(self.nbr_models)]
        self.output_layer = output_layer
        
    def forward(self, inputs):
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        
        if self.input_processing_layer is not None:
            inputs = self.input_processing_layer(inputs)
        
        first_embedding, first_prior_masks_list, first_states = self.models_list[0].forward(inputs)
        embeddings_list = [first_embedding]
        prior_masks_list = [first_prior_masks_list]
        states_list = [first_states]
        
        for model in self.models_list[1:]:
            tmp_embedding, tmp_prior_masks_list, tmp_states = model.forward(inputs)
            embeddings_list.append(tmp_embedding)
            prior_masks_list.append(tmp_prior_masks_list)
            states_list.append(tmp_states)
        
        output = self.output_layer(tf.keras.layers.Concatenate()(embeddings_list))
        
        return output, prior_masks_list, states_list
    
    def call(self, inputs):
        return self.forward(inputs)[0]
    
    def masks_explain(self, inputs):
        return self.forward(inputs)[1]
    
    def final_states(self, inputs):
        return self.forward(inputs)[2]