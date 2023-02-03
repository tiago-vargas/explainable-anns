# For type annotations
from docplex.mp.model import Model
from docplex.mp.constr import LinearConstraint
from docplex.mp.dvar import Var


def stretch_interval_to_the_right(model: Model, constraint: LinearConstraint, step: float) -> None:
    model.solve()

    x: Var = constraint.left_expr
    while model.solution is not None and constraint.right_expr.constant <= x.ub:
        constraint.right_expr.constant += step
        model.solve()
