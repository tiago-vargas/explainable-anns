import numpy as np
from docplex.mp.model import Model
from keras.models import Sequential


class MILPModel:
    def __init__(self, network: Sequential):
        self._codify_model(network)

    def _codify_model(self, network: Sequential):
        self._model = Model()
        input_size = network.input_shape[1]
        i = self._model.continuous_var_list(keys=input_size, name='i')
        output_size = network.output_shape[1]
        o = self._model.continuous_var_list(keys=output_size, name='o')
        s_output = self._model.continuous_var_list(keys=output_size, name='s(o)')

        if len(network.layers) == 2:
            layer = network.layers[0]
            w = layer.weights[0].numpy()
            b = layer.weights[1].numpy()

            layer_size = network.layers[0].units
            x = self._model.continuous_var_list(keys=layer_size, name='x(0)')
            s = self._model.continuous_var_list(keys=layer_size, name='s(0)')

        output_layer = network.layers[-1]
        weights = output_layer.weights[0].numpy()
        biases = output_layer.weights[1].numpy()

        i = np.array(i)
        there_are_no_hidden_layers = (len(network.layers) == 1)
        if there_are_no_hidden_layers:
            for j in range(output_size):
                self._model.add_constraint(weights.T[j, :] @ i + biases[j] == o[j] - s_output[j])
        else:
            for j in range(output_size):
                self._model.add_constraint(w.T[0, :] @ i + b[0] == x[0] - s[0])
                self._model.add_constraint(weights.T[j, :] @ x + biases[j] == o[j] - s_output[j])

        z_output = self._model.binary_var_list(keys=output_size, name='z(o)')
        for j in range(output_size):
            self._model.add_indicator(binary_var=z_output[j], active_value=1, linear_ct=(o[j] <= 0))
            self._model.add_indicator(binary_var=z_output[j], active_value=0, linear_ct=(s_output[j] <= 0))

        if len(network.layers) == 2:
            z = self._model.binary_var_list(keys=layer_size, name='z(0)')
            for j in range(layer_size):
                self._model.add_indicator(binary_var=z[j], active_value=1, linear_ct=(x[j] <= 0))
                self._model.add_indicator(binary_var=z[j], active_value=0, linear_ct=(s[j] <= 0))

    @property
    def formulation(self):
        return list(self._model.iter_constraints())
