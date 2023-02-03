from docplex.mp.model import Model
import slow_and_steady_stretcher


class TestImprovingExplanation:
    class TestStretchingInterval:
        class TestSlowAndSteadyMethod:
            def test_stretching_to_the_right(self):
                model = Model()
                x = model.continuous_var(lb=-3, ub=4, name='x')

                # Our objective is to stretch the constraint below as far to the right as we can
                constraint = model.add_constraint(x <= 0)
                model.maximize(x)

                slow_and_steady_stretcher.stretch_interval_to_the_right(model, constraint, step=0.5)

                model.solve()
                assert x.solution_value == 4

            def test_stretching_to_the_right_with_other_constraints(self):
                model = Model()
                x = model.continuous_var(lb=-3, ub=4, name='x')
                model.add_constraint(x <= 3)

                # Our objective is to stretch the constraint below as far to the right as we can
                constraint = model.add_constraint(x <= 0)
                model.maximize(x)

                slow_and_steady_stretcher.stretch_interval_to_the_right(model, constraint, step=0.1)

                model.solve()
                assert x.solution_value == 3
