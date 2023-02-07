# For type annotations
from docplex.mp.dvar import Var
from docplex.mp.model import Model
from docplex.mp.constr import LinearConstraint


def stretch_to_the_right(model: Model, constraint: LinearConstraint):
    x: Var = constraint.left_expr
    constraint.right_expr.constant = x.ub
    model.maximize(x)
    model.solve()

    constraint.right_expr.constant = x.solution_value

    model.remove_objective()
