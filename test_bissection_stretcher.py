from docplex.mp.model import Model

import bissection_stretcher


class TestStretcher:
    class TestStretchingToTheRight:
        def test_base_on_bounds_alone(self):
            model = Model()
            x = model.continuous_var(lb=0, ub=5, name='x')
            constraint = model.add_constraint(x <= 0, ctname='constraint to stretch')

            bissection_stretcher.stretch_to_the_right(model, constraint)

            assert constraint.right_expr.constant == 5

        def test_base_on_another_constraint(self):
            model = Model()
            x = model.continuous_var(lb=0, ub=5, name='x')
            model.add_constraint(x <= 3)
            constraint = model.add_constraint(x <= 0, ctname='constraint to stretch')

            bissection_stretcher.stretch_to_the_right(model, constraint)

            assert constraint.right_expr.constant == 3

