from docplex.mp.dvar import Var
from docplex.mp.model import Model
from keras.layers import Dense
from keras.models import Sequential


class MILPModel:
    def __init__(self, network: Sequential):
        self._network = network
        self._codify_model()

    def _codify_model(self):
        """
        Codifies a MILP Model from the network and stores it in `self.formulation`.
        """
        def create_and_add_variables_for_all_units():
            def create_and_add_variables_for_input_units():
                input_size = self._network.input_shape[1]
                self._model.continuous_var_list(keys=input_size, name='i')

            def create_and_add_variables_for_all_hidden_units():
                for layer in self._hidden_layers:
                    def create_and_add_hidden_layer_variables():
                        layer_index = self._network.layers.index(layer)
                        layer_size = self._network.layers[layer_index].units
                        self._model.continuous_var_list(keys=layer_size, name='x(%d)' % layer_index)

                    create_and_add_hidden_layer_variables()

            def create_and_add_variables_for_output_units():
                output_size = self._network.output_shape[1]
                self._model.continuous_var_list(keys=output_size, name='o')

            create_and_add_variables_for_input_units()
            create_and_add_variables_for_all_hidden_units()
            create_and_add_variables_for_output_units()

        def create_and_add_constraints_for_the_connections_using_relu_activation():
            def create_and_add_slack_variables_for_all_hidden_layers():
                for layer in self._hidden_layers:
                    def create_and_add_slack_variables_for_hidden_layer():
                        layer_index = self._network.layers.index(layer)
                        layer_size = layer.units
                        self._model.continuous_var_list(keys=layer_size, name='s(%d)' % layer_index)

                    create_and_add_slack_variables_for_hidden_layer()

            def create_and_add_slack_variables_for_the_output_layer():
                output_size = self._network.output_shape[1]
                self._model.continuous_var_list(keys=output_size, name='s(o)')

            def _find_layer_slack_variables(layer_index: int) -> list[Var]:
                is_last_layer = (layer_index == len(self._network.layers) - 1)
                if is_last_layer:
                    result = self._model.find_matching_vars('s(o)')
                else:
                    result = self._model.find_matching_vars('s(%d)' % layer_index)
                return result

            def _find_layer_units(layer_index: int) -> list[Var]:
                is_last_layer = (layer_index == len(self._network.layers) - 1)
                if is_last_layer:
                    result = self._model.find_matching_vars('o')
                else:
                    result = self._model.find_matching_vars('x(%d)' % layer_index)
                return result

            def add_indicators_for_the_hidden_layer(layer: Dense):
                layer_index = self._network.layers.index(layer)
                units = _find_layer_units(layer_index)
                slack_variables = _find_layer_slack_variables(layer_index)
                layer_size = layer.units
                z = self._model.binary_var_list(keys=layer_size, name='z(%d)' % layer_index)
                for i in range(layer_size):
                    self._model.add_indicator(binary_var=z[i], active_value=1, linear_ct=(units[i] <= 0))
                    self._model.add_indicator(binary_var=z[i], active_value=0, linear_ct=(slack_variables[i] <= 0))

            def add_indicators_for_the_output_layer():
                output_units = self._model.find_matching_vars('o')
                slack_variables = self._model.find_matching_vars('s(o)')
                output_size = self._network.output_shape[1]
                z = self._model.binary_var_list(keys=output_size, name='z(o)')
                for i in range(output_size):
                    self._model.add_indicator(binary_var=z[i], active_value=1, linear_ct=(output_units[i] <= 0))
                    self._model.add_indicator(binary_var=z[i], active_value=0, linear_ct=(slack_variables[i] <= 0))

            def _add_constraints_describing_connections(layer: Dense):
                """
                Adds constraints to `self._model` describing connections from all units of this `layer` and all the
                units from the previous layer.
                """
                def _add_constraint_describing_unit(unit: Var):
                    """
                    Adds constraints to `self._model` describing connections from this `unit` and all the units from the previous
                    layer.
                    """

                    def get_index_of_unit() -> int:
                        return int(unit.name.split('_')[-1])

                    def find_previous_layer_units(layer_index: int) -> list[Var]:
                        is_first_hidden_layer = (layer_index == 0)
                        if is_first_hidden_layer:
                            result = self._model.find_matching_vars('i')
                        else:
                            result = self._model.find_matching_vars('x(%d)' % (layer_index - 1))
                        return result

                    def find_layer_index_of_unit() -> int:
                        is_output_variable = (unit.name.startswith('o_'))
                        if is_output_variable:
                            result = len(self._network.layers) - 1
                        else:
                            # if '(' is in unit.name...
                            result = int(unit.name[unit.name.find('(') + 1:unit.name.find(')')])
                        return result

                    layer_index = find_layer_index_of_unit()
                    previous_layer_units = find_previous_layer_units(layer_index)

                    unit_index = get_index_of_unit()

                    layer = self._network.layers[layer_index]
                    biases = layer.weights[1].numpy()
                    bias = biases[unit_index]

                    slack_variable = _find_layer_slack_variables(layer_index)[unit_index]
                    weights = layer.weights[0].numpy()[:, unit_index]
                    self._model.add_constraint(weights.T @ previous_layer_units + bias == unit - slack_variable)

                i = self._network.layers.index(layer)
                layer_units = _find_layer_units(i)

                is_last_layer = (i == len(self._network.layers) - 1)
                if is_last_layer:
                    output_size = self._network.output_shape[1]
                    for j in range(output_size):
                        _add_constraint_describing_unit(layer_units[j])
                else:
                    layer_size = layer.units
                    for j in range(layer_size):
                        _add_constraint_describing_unit(layer_units[j])

            create_and_add_slack_variables_for_all_hidden_layers()
            create_and_add_slack_variables_for_the_output_layer()
            for layer in self._hidden_layers:
                _add_constraints_describing_connections(layer)
                add_indicators_for_the_hidden_layer(layer)
            output_layer = self._network.layers[-1]
            _add_constraints_describing_connections(output_layer)
            add_indicators_for_the_output_layer()

        self._model = Model()
        create_and_add_variables_for_all_units()
        create_and_add_constraints_for_the_connections_using_relu_activation()

    @property
    def formulation(self):
        return self._model.iter_constraints()

    @property
    def _hidden_layers(self):
        return self._network.layers[:-1]
