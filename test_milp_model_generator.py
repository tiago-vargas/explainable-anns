import numpy as np
from keras.layers import Dense
from keras.models import Sequential


def test_codifying_3x2x1_network():
    model = Sequential()
    model.add(Dense(units=2, input_dim=3))
    model.add(Dense(units=1))
    weights_0 = np.array([[11, 21], [12, 22], [13, 23]])
    biases_0 = np.array([10, 20])
    model.layers[0].set_weights([weights_0, biases_0])
    weights_1 = np.array([[1], [2]])
    biases_1 = np.array([30])
    model.layers[1].set_weights([weights_1, biases_1])

    milp_model = MILPModel(model)

    assert milp_model.formulation == []
