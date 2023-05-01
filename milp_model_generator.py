import numpy as np
from docplex.mp.dvar import Var
from docplex.mp.model import Model
from keras.models import Sequential


class MILPModel:
    def __init__(self, network: Sequential):
        self._codify_model(network)

    def _codify_model(self, network: Sequential):
        self._model = Model()

        i = _create_and_add_input_variables(network, self._model)
        o = _create_and_add_output_variables(network, self._model)
        s_output = _create_and_add_output_slack_variables(network, self._model)

        output_layer = network.layers[-1]
        weights = output_layer.weights[0].numpy()
        biases = output_layer.weights[1].numpy()

        i = np.array(i)
        output_size = network.output_shape[1]
        there_are_no_hidden_layers = (len(network.layers) == 1)
        if there_are_no_hidden_layers:
            for j in range(output_size):
                self._model.add_constraint(weights.T[j, :] @ i + biases[j] == o[j] - s_output[j])
        else:
            layer = network.layers[0]
            w = layer.weights[0].numpy()
            b = layer.weights[1].numpy()

            x = _create_and_add_hidden_layer_variables(network, self._model)
            s = _create_and_add_hidden_layer_slack_variables(network, self._model)

            for j in range(output_size):
                self._model.add_constraint(w.T[0, :] @ i + b[0] == x[0] - s[0])
                self._model.add_constraint(weights.T[j, :] @ x + biases[j] == o[j] - s_output[j])

            self._add_indicators_for_the_hidden_layer(network, s, x)

        self._add_indicators_for_the_output_layer(o, s_output, network)

    def _add_indicators_for_the_hidden_layer(self, network: Sequential, slack_variables, x):
        layer_size = network.layers[0].units
        z = self._model.binary_var_list(keys=layer_size, name='z(0)')
        for j in range(layer_size):
            self._model.add_indicator(binary_var=z[j], active_value=1, linear_ct=(x[j] <= 0))
            self._model.add_indicator(binary_var=z[j], active_value=0, linear_ct=(slack_variables[j] <= 0))

    def _add_indicators_for_the_output_layer(self, output_variables, slack_variables, network: Sequential):
        output_size = network.output_shape[1]
        z = self._model.binary_var_list(keys=output_size, name='z(o)')
        for i in range(output_size):
            self._model.add_indicator(binary_var=z[i], active_value=1, linear_ct=(output_variables[i] <= 0))
            self._model.add_indicator(binary_var=z[i], active_value=0, linear_ct=(slack_variables[i] <= 0))

    @property
    def formulation(self):
        return self._model.iter_constraints()


def _create_and_add_input_variables(network: Sequential, model: Model) -> list[Var]:
    input_size = network.input_shape[1]
    input_variables = model.continuous_var_list(keys=input_size, name='i')
    return input_variables


# TODO: generalize to work for any layer, not just layer 0
def _create_and_add_hidden_layer_variables(network: Sequential, model: Model) -> list[Var]:
    layer_size = network.layers[0].units
    hidden_layer_variables = model.continuous_var_list(keys=layer_size, name='x(0)')
    return hidden_layer_variables


# TODO: generalize to work for any layer, not just layer 0
def _create_and_add_hidden_layer_slack_variables(network: Sequential, model: Model) -> list[Var]:
    layer_size = network.layers[0].units
    slack_variables = model.continuous_var_list(keys=layer_size, name='s(0)')
    return slack_variables


def _create_and_add_output_variables(network: Sequential, model: Model) -> list[Var]:
    output_size = network.output_shape[1]
    output_variables = model.continuous_var_list(keys=output_size, name='o')
    return output_variables


def _create_and_add_output_slack_variables(network, model: Model) -> list[Var]:
    output_size = network.output_shape[1]
    slack_variables = model.continuous_var_list(keys=output_size, name='s(o)')
    return slack_variables
