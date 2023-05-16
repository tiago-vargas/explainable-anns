from keras import Sequential
from keras.layers import Dense

from explainer import Explainer
from milp_model_generator import MILPModel


class TestMinimalExplanation:
    def test_identifying_relevant_features(self):
        # fake_predict = lambda x: 0 if x[0] >= 0 else 1
        def fake_predict(inputs: list[float]) -> int:
            x = inputs[0]
            if x >= 0:
                return 0
            else:
                return 1
        nn = Sequential()
        nn.add(Dense(input_dim=1, units=2))
        m = MILPModel(nn)
        explainer = Explainer(milp_model=m, predicting_function=fake_predict, feature_labels=['A'])

        relevant_features = explainer.get_relevant_features(input=[0.7])

        assert relevant_features == ['A']
