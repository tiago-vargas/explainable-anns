from docplex.mp.dvar import Var
from docplex.mp.model import Model
from keras.layers import Dense
from keras.models import Sequential


class MILPModel:
    def __init__(self, network: Sequential):
        self._codify_model(network)

    def _codify_model(self, network: Sequential):
        """
        Codifies a MILP Model from the network and stores it in `self.formulation`.
        """
        self._model = Model()
        self._create_and_add_variables_for_all_units(network)
        self._create_and_add_constraints_for_the_connections_using_relu_activation(network)

    def _create_and_add_variables_for_all_units(self, network):
        self._create_and_add_variables_for_input_units(network)
        self._create_and_add_variables_for_all_hidden_units(network)
        self._create_and_add_variables_for_output_units(network)

    def _create_and_add_variables_for_input_units(self, network: Sequential):
        input_size = network.input_shape[1]
        self._model.continuous_var_list(keys=input_size, name='i')

    def _create_and_add_variables_for_all_hidden_units(self, network: Sequential):
        hidden_layers = network.layers[:-1]
        for layer in hidden_layers:
            _create_and_add_hidden_layer_variables(network, self._model, layer)

    def _create_and_add_variables_for_output_units(self, network: Sequential):
        output_size = network.output_shape[1]
        self._model.continuous_var_list(keys=output_size, name='o')

    def _create_and_add_constraints_for_the_connections_using_relu_activation(self, network):
        self._create_and_add_slack_variables_for_all_hidden_layers(network)
        self._create_and_add_slack_variables_for_the_output_layer(network)
        hidden_layers = network.layers[:-1]
        for layer in hidden_layers:
            self._add_constraints_describing_connections(network, layer)
            self._add_indicators_for_the_hidden_layer(network, layer)
        output_layer = network.layers[-1]
        self._add_constraints_describing_connections(network, output_layer)
        self._add_indicators_for_the_output_layer(network)

    def _create_and_add_slack_variables_for_all_hidden_layers(self, network):
        hidden_layers = network.layers[:-1]
        for layer in hidden_layers:
            self._create_and_add_slack_variables_for_hidden_layer(network, layer)

    def _create_and_add_slack_variables_for_hidden_layer(self, network: Sequential, layer: Dense):
        layer_index = network.layers.index(layer)
        layer_size = layer.units
        self._model.continuous_var_list(keys=layer_size, name='s(%d)' % layer_index)

    def _create_and_add_slack_variables_for_the_output_layer(self, network: Sequential):
        output_size = network.output_shape[1]
        self._model.continuous_var_list(keys=output_size, name='s(o)')

    def _add_constraints_describing_connections(self, network: Sequential, layer: Dense):
        """
        Adds constraints to `self._model` describing connections from all units of this `layer` and all the units from
        the previous layer.
        """
        i = network.layers.index(layer)
        layer_units = self._find_layer_units(network, i)

        is_last_layer = (i == len(network.layers) - 1)
        if is_last_layer:
            output_size = network.output_shape[1]
            for j in range(output_size):
                self._add_constraint_describing_unit(network, layer_units[j])
        else:
            layer_size = layer.units
            for j in range(layer_size):
                self._add_constraint_describing_unit(network, layer_units[j])

    def _find_layer_slack_variables(self, network: Sequential, layer_index: int) -> list[Var]:
        is_last_layer = (layer_index == len(network.layers) - 1)
        if is_last_layer:
            slack_variables = self._model.find_matching_vars('s(o)')
        else:
            slack_variables = self._model.find_matching_vars('s(%d)' % layer_index)
        return slack_variables

    def _find_layer_units(self, network: Sequential, layer_index: int) -> list[Var]:
        is_last_layer = (layer_index == len(network.layers) - 1)
        if is_last_layer:
            layer_units = self._model.find_matching_vars('o')
        else:
            layer_units = self._model.find_matching_vars('x(%d)' % layer_index)
        return layer_units

    def _find_previous_layer_units(self, layer_index: int) -> list[Var]:
        is_first_hidden_layer = (layer_index == 0)
        if is_first_hidden_layer:
            previous_layer_units = self._model.find_matching_vars('i')
        else:
            previous_layer_units = self._model.find_matching_vars('x(%d)' % (layer_index - 1))
        return previous_layer_units

    def _add_constraint_describing_unit(self, network: Sequential, unit: Var):
        """
        Adds constraints to `self._model` describing connections from this `unit` and all the units from the previous
        layer.
        """
        layer_index = _find_layer_of_unit(network, unit)
        previous_layer_units = self._find_previous_layer_units(layer_index)

        unit_index = _get_index_of_unit(unit)

        layer = network.layers[layer_index]
        biases = layer.weights[1].numpy()
        bias = biases[unit_index]

        slack_variable = self._find_layer_slack_variables(network, layer_index)[unit_index]
        weights = layer.weights[0].numpy()[:, unit_index]
        self._model.add_constraint(weights.T @ previous_layer_units + bias == unit - slack_variable)

    def _add_indicators_for_the_hidden_layer(self, network: Sequential, layer: Dense):
        layer_index = network.layers.index(layer)
        units = self._find_layer_units(network, layer_index)
        slack_variables = self._find_layer_slack_variables(network, layer_index)
        layer_size = layer.units
        z = self._model.binary_var_list(keys=layer_size, name='z(%d)' % layer_index)
        for i in range(layer_size):
            self._model.add_indicator(binary_var=z[i], active_value=1, linear_ct=(units[i] <= 0))
            self._model.add_indicator(binary_var=z[i], active_value=0, linear_ct=(slack_variables[i] <= 0))

    def _add_indicators_for_the_output_layer(self, network: Sequential):
        output_units = self._model.find_matching_vars('o')
        slack_variables = self._model.find_matching_vars('s(o)')
        output_size = network.output_shape[1]
        z = self._model.binary_var_list(keys=output_size, name='z(o)')
        for i in range(output_size):
            self._model.add_indicator(binary_var=z[i], active_value=1, linear_ct=(output_units[i] <= 0))
            self._model.add_indicator(binary_var=z[i], active_value=0, linear_ct=(slack_variables[i] <= 0))

    @property
    def formulation(self):
        return self._model.iter_constraints()


def _create_and_add_hidden_layer_variables(network: Sequential, model: Model, layer):
    layer_index = network.layers.index(layer)
    layer_size = network.layers[layer_index].units
    model.continuous_var_list(keys=layer_size, name='x(%d)' % layer_index)


def _get_index_of_unit(unit: Var) -> int:
    return int(unit.name.split('_')[-1])


def _find_layer_of_unit(network: Sequential, unit: Var) -> int:
    is_output_variable = (unit.name.startswith('o_'))
    if is_output_variable:
        layer_index = len(network.layers) - 1
    else:
        # if '(' is in unit.name...
        layer_index = int(unit.name[unit.name.find('(') + 1:unit.name.find(')')])
    return layer_index
