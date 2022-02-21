import tensorflow as tf
import keras_tuner as kt
import numpy as np
import pandas as pd
import math


class HyperModelCrossValidation(kt.HyperModel):
    """A HyperModel class that performe cross validation.
    Note: For cross validation, we build several time a model per trial -> TensorBoard does not work"""
    
    def __init__(self,
                 build_model,
                 nbr_folds,
                 batch_sizes_to_try_list,
                 optimization_metric='loss',
                 minimize_optimization_metric=True,
                 hyper_callbacks=None,
                 **kwargs):
        if nbr_folds < 2:
            raise ValueError(f"'nbr_folds' needs to be greater or equal to 2 for cross validation. Received: {nbr_folds}")
        super().__init__(**kwargs)
        self.build_model = build_model
        self.nbr_folds = nbr_folds
        self.batch_sizes_to_try_list = batch_sizes_to_try_list
        self.optimization_metric = optimization_metric
        self.minimize_optimization_metric = minimize_optimization_metric
        self.hyper_callbacks = hyper_callbacks
        
    def build(self, hp):
        return self.build_model(hp)
    
    def fit(self, hp, model, x, y, *args, **kwargs):
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(x, pd.DataFrame):
            x = x.values
        elif isinstance(x, dict):
            for feature_key, feature_values in x.items():
                if isinstance(feature_values, pd.Series):
                    x[feature_key] = feature_values.values
        
        if isinstance(y, type(list)) or isinstance(y, np.ndarray):
            nbr_samples = len(y)
            folds_size = math.ceil(nbr_samples/self.nbr_folds)
        else:
            raise ValueError(f"'y' type expected: list/np.ndarray/pd.Series. Received: {type(y)}")
        
        tmp_callbacks = kwargs['callbacks']
        
        for fold_id, lower_bound in enumerate(np.arange(0, nbr_samples-1, folds_size)):
            print(f"-- FOLD NUMBER: {fold_id} --")
            
            tmp_model = self.build(hp)
            instantiated_hyper_callbacks = [hyper_callback(hp) for hyper_callback in self.hyper_callbacks]
            kwargs['callbacks'] = [*tmp_callbacks, *instantiated_hyper_callbacks]
            
            if isinstance(x, type(list)) or isinstance(x, np.ndarray):
                x_train = [*x[:lower_bound], *x[lower_bound+folds_size:]]
                x_val = x[lower_bound:lower_bound+folds_size]
            elif isinstance(x, dict):
                x_train = {k:[*v[:lower_bound], *v[lower_bound+folds_size:]] for k, v in x.items()}
                x_val = {k:v[lower_bound:lower_bound+folds_size] for k, v in x.items()}
            else:
                raise ValueError(f"'x' type expected: list/np.ndarray/pd.DataFrame/dict. Received: {type(x)}")
            y_train = [*y[:lower_bound], *y[lower_bound+folds_size:]]
            y_val = y[lower_bound:lower_bound+folds_size]
            
            batch_size = hp.Choice('batch_size', self.batch_sizes_to_try_list)
            train_tensor = tf.data.Dataset.from_tensor_slices((x_train,
                                                               y_train)).batch(batch_size)
            validation_tensor = tf.data.Dataset.from_tensor_slices((x_val,
                                                                    y_val)).batch(batch_size)
            history = tmp_model.fit(train_tensor,
                                    validation_data=validation_tensor,
                                    *args,
                                    **kwargs)
            
            if self.minimize_optimization_metric:
                best_epoch = np.argmin(history.history[self.optimization_metric])
            else:
                best_epoch = np.argmax(history.history[self.optimization_metric])
            if fold_id == 0:
                return_metrics = {k:v[best_epoch] for k, v in history.history.items()}
            else:
                for k, v in history.history.items():
                    return_metrics[k] += v[best_epoch]
        
        return {k: v/self.nbr_folds for k, v in return_metrics.items()}