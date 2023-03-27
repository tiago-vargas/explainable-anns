from docplex.mp.model import Model

import bissection_stretcher


class TestStretcher:
    class TestStretchingToTheRight:
        # def test_base_on_bounds_alone(self):
        #     model = Model()
        #     x = model.continuous_var(lb=0, ub=5, name='x')
        #     constraint = model.add_constraint(x <= 0, ctname='constraint to stretch')
        #
        #     bissection_stretcher.stretch_to_the_right(model, constraint)
        #
        #     assert constraint.right_expr.constant == 5
        #
        # def test_base_on_another_constraint(self):
        #     model = Model()
        #     x = model.continuous_var(lb=0, ub=5, name='x')
        #     model.add_constraint(x <= 3)
        #     constraint = model.add_constraint(x <= 0, ctname='constraint to stretch')
        #
        #     bissection_stretcher.stretch_to_the_right(model, constraint)
        #
        #     assert constraint.right_expr.constant == 3

        def test_detecting_gaps(self):
            model = Model()
            # x \in [0, 9) U (12, 20]
            x = model.continuous_var(lb=0, ub=20, name='x')
            y = model.continuous_var(lb=0, ub=20, name='y')
            b_1 = model.binary_var(name='b_1')
            b_2 = model.binary_var(name='b_2')
            model.add_indicator(b_1, x >= 12)
            model.add_indicator(b_2, x <= 9)
            # I want `b_1 == True or b_2 == True`
            model.add_constraint(b_1 + b_2 >= 1)
            precision = 0.01

            constraint = model.add_constraint(x <= 0, ctname='constraint to stretch')

            bissection_stretcher.stretch_to_the_right(model, constraint, precision)

            assert 10 - precision <= constraint.right_expr.constant <= 10 + precision
