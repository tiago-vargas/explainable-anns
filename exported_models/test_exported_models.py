import pandas as pd
import tensorflow as tf
from milp import codify_network


class TestMilpModel:
    def test_my_model_equals_levi_model_for_glass(self):
        dataset = 'glass'
        model = tf.keras.models.load_model(f'../datasets/{dataset}/teste.h5')

        data_test = pd.read_csv(f'../datasets/{dataset}/test.csv')
        data_train = pd.read_csv(f'../datasets/{dataset}/train.csv')
        data = data_train.append(data_test)
        data = data[['RI', 'Na', 'target']]

        mdl, _ = codify_network(model, data, 'tjeng', False)

        with open('original_milp_mdl_for_glass.lp', 'r') as file:
            file_as_str = str(file.read())

        assert mdl.export_to_string() == file_as_str
