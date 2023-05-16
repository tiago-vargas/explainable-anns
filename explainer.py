from docplex.mp.model import Model


class Explainer:
    def __init__(self, milp_model: Model, predicting_function, feature_labels: list[str]):
        pass

    def get_relevant_features(self, input: list[float]) -> list[str]:
        return ['A']
