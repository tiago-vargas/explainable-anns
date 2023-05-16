from milp_model_generator import MILPModel


class Explainer:
    def __init__(self, milp_model: MILPModel, predicting_function, feature_labels: list[str]):
        self.feature_labels = feature_labels

    def get_relevant_features(self, input: list[float]) -> list[str]:
        return self.feature_labels
