import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from milp_model_generator import MILPModel


class TestFormulatingNetwork:
    class TestWithoutHiddenLayers:
        def test_1x1_network(self):
            nn = Sequential()
            nn.add(Dense(units=1, input_dim=1))
            weights = np.array([[11]])
            biases = np.array([50])
            nn.layers[0].set_weights([weights, biases])

            m = MILPModel(nn)

            strings = {x.to_string() for x in m.formulation}
            assert strings == {
                '11i_0+50 == o_0-s(o)_0',
                'z(o)_0 -> [o_0 <= 0]',
                'z(o)_0=0 -> [s(o)_0 <= 0]',
            }

        def test_2x1_network(self):
            nn = Sequential()
            nn.add(Dense(units=1, input_dim=2))
            weights = np.array([[11], [12]])
            biases = np.array([50])
            nn.layers[0].set_weights([weights, biases])

            m = MILPModel(nn)

            strings = {x.to_string() for x in m.formulation}
            assert strings == {
                '11i_0+12i_1+50 == o_0-s(o)_0',
                'z(o)_0 -> [o_0 <= 0]',
                'z(o)_0=0 -> [s(o)_0 <= 0]',
            }

        def test_1x2_network(self):
            nn = Sequential()
            nn.add(Dense(units=2, input_dim=1))
            weights = np.array([[11, 21]])
            biases = np.array([50, 70])
            nn.layers[0].set_weights([weights, biases])

            m = MILPModel(nn)

            strings = {x.to_string() for x in m.formulation}
            assert strings == {
                '11i_0+50 == o_0-s(o)_0',
                'z(o)_0 -> [o_0 <= 0]',
                'z(o)_0=0 -> [s(o)_0 <= 0]',

                '21i_0+70 == o_1-s(o)_1',
                'z(o)_1 -> [o_1 <= 0]',
                'z(o)_1=0 -> [s(o)_1 <= 0]',
            }
