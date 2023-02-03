from docplex.mp.model import Model
from docplex.mp.constr import LinearConstraint


def stretch_to_the_right(model: Model, constraint: LinearConstraint):
    var = constraint.left_expr
    constraint.right_expr = var.ub
