r"""
# Optimization

We want to minimize the Rosenbrock function $f(x,y)=(1-x)^2+100(y-x**2)^2$
over the domain $[-2,2]^2$.
"""
from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario

# %%
# Before starting,
# we activate the logger as an optimization process logs meaningful information.
configure_logger()

# %%
# Firstly,
# we define the discipline computing the Rosenbrock function
# and the Euclidean distance to the optimum:
discipline = create_discipline(
    "AnalyticDiscipline",
    expressions={"z": "(1-x)**2+100*(y-x**2)**2", "c": "((x-1)**2+(y-1)**2)**0.5"},
    name="Rosenbrock"
)

# %%
# Then, we create the design space:
design_space = create_design_space()
design_space.add_variable("x", l_b=-2., u_b=2., value=0.)
design_space.add_variable("y", l_b=-2., u_b=2., value=0.)

# %%
# Thirdly,
# we put these elements together in a scenario
# to minimize the Rosenbrock function
# under the constraint that the distance
# between the design point and the solution of the unconstrained problem
# is greater or equal to 1.
scenario = create_scenario([discipline], "DisciplinaryOpt", "z", design_space)
scenario.add_constraint("c", constraint_type="ineq", positive=True, value=1.)

# %%
# !!! note
#
#     GEMSEO is a Python library
#     dedicated to multidisciplinary design optimization (MDO)
#     based on the notion of MDO formulation.
#     This is why the second positional argument `formulation` is mandatory.
#     But when using the scenario with a unique discipline,
#     don't bother and consider `"DisciplinaryOpt"`.

# %%
# before executing it with a gradient-free optimizer:
scenario.execute({"algo": "NLOPT_COBYLA", "max_iter": 100})

# %%
# Lastly,
# we can plot the optimization history:
scenario.post_process("OptHistoryView", save=False, show=True)
