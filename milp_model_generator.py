import numpy as np
from docplex.mp.model import Model
from keras.models import Sequential


class MILPModel:
    def __init__(self, network: Sequential):
        self._codify_model(network)

    def _codify_model(self, network: Sequential):
        self._model = Model()
        layer = network.layers[0]
        weights = layer.weights[0].numpy()
        biases = layer.weights[1].numpy()
        input_size = network.input_shape[1]
        i = self._model.continuous_var_list(keys=input_size, name='i')
        output_size = network.output_shape[1]
        o = self._model.continuous_var_list(keys=output_size, name='o')
        s = self._model.continuous_var_list(keys=output_size, name='s')

        i = np.array(i)
        for j in range(output_size):
            self._model.add_constraint(weights.T[j, :] @ i + biases[j] == o[j] - s[j])

        z = self._model.binary_var_list(keys=output_size, name='z')
        for j in range(output_size):
            self._model.add_indicator(binary_var=z[j], active_value=1, linear_ct=(o[j] <= 0))
            self._model.add_indicator(binary_var=z[j], active_value=0, linear_ct=(s[j] <= 0))

    @property
    def formulation(self):
        return list(self._model.iter_constraints())
