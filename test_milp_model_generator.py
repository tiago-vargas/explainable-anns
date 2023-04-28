import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from milp_model_generator import MILPModel


class TestFormulatingNetwork:
    def test_1x1_network(self):
        nn = Sequential()
        nn.add(Dense(units=1, input_dim=1))
        weights = np.array([[11]])
        biases = np.array([50])
        nn.layers[0].set_weights([weights, biases])

        m = MILPModel(nn)

        strings = [x.to_string() for x in m.formulation]
        assert strings == [
            '11i+50 == o-s',
            'z -> [o <= 0]',
            'z=0 -> [s <= 0]',
        ]
