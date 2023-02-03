# For type annotations
from docplex.mp.model import Model
from docplex.mp.constr import LinearConstraint
from docplex.mp.dvar import Var


def stretch_interval_to_the_right(model: Model, constraint: LinearConstraint, step: float) -> None:
    var: Var = constraint.left_expr
    model.remove_constraint(constraint)
    constraint = var <= var.ub
    model.add_constraint(constraint)
