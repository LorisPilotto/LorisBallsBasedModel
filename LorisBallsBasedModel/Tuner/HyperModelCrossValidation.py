import keras_tuner as kt
import numpy as np
import math


class HyperModelCrossValidation(kt.HyperModel):
    """A HyperModel class that performe cross validation."""
    
    def __init__(self,
                 build_model,
                 nbr_folds,
                 batch_size_to_try_list,
                 optimization_metric='loss',
                 minimize_optimization_metric=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.build_model = build_model
        self.nbr_folds = nbr_folds
        self.batch_size_to_try_list = batch_size_to_try_list
        self.optimization_metric = optimization_metric
        self.minimize_optimization_metric = minimize_optimization_metric
        
    def build(self, hp):
        return self.build_model(hp)
    
    def fit(self, hp, model, x, y, *args, **kwargs):
        iterate and separate data in train val -> call the fit methode from the super (first iteration if it output a list or a dict act accordingly so that it has list(s))
        if isinstance(y, list) or isinstance(y, np.array):
            nbr_samples = len(y)
            folds_size = math.ceil(nbr_samples/self.nbr_folds)
        else:
            raise ValueError(f"'y' type expected: list/np.array. Received: {type(y)}")
            
        for lower_bound in np.arange(0, nbr_samples-1, folds_size):
            if isinstance(x, list) or isinstance(x, np.array) or isinstance(x, pd.DataFrame):
                x_train = [*x[:lower_bound], *x[lower_bound+folds_size:]]
                x_val = x[lower_bound:lower_bound+fold_size]
            elif isinstance(x, dict):
                x_train = {k:[*v[:lower_bound], *v[lower_bound+folds_size:]] for k, v in x.items()}
                x_val = {k:v[lower_bound:lower_bound+fold_size] for k, v in x.items()}
            else:
                raise ValueError(f"'x' type expected: list/np.array/dict/pd.DataFrame. Received: {type(x)}")
            y_train = [*y[:lower_bound], *y[lower_bound+folds_size:]]
            y_val = y[lower_bound:lower_bound+fold_size]
            
            batch_size = hp.Choice('batch_size', self.batch_size_to_try_list)
            train_tensor = tf.data.Dataset.from_tensor_slices((x_train,
                                                               y_train)).batch(batch_size)
            validation_tensor = tf.data.Dataset.from_tensor_slices((x_val,
                                                                    y_val)).batch(batch_size)
            history = model.fit(train_tensor,
                                validation_data=validation_tensor,
                                *args,
                                **kwargs)
            
            if self.minimize_optimization_metric:
                best_epoch = np.argmin(history.history[self.optimization_metric])
            else:
                best_epoch = np.argmax(history.history[self.optimization_metric])
            if lower_bound == 0:
                return_metrics = {k:v[best_epoch] for k, v in history.history.items()}
            else:
                {return_metrics[k] += v[best_epoch] for k, v in history.history.items()}
        
        return {k: v/self.nbr_folds for k, v in return_metrics.items()}
            