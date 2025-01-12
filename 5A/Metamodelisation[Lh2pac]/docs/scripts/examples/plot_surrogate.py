r"""
# Surrogate model

In this example,
we will build a surrogate model of the Rosenbrock function
and a constraint related to an Rosenbrock-based optimization problem.
"""

import pickle
from pathlib import Path

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_surrogate
from gemseo import import_discipline
from gemseo.mlearning.quality_measures.r2_measure import R2Measure
from gemseo.mlearning.quality_measures.rmse_measure import RMSEMeasure
from gemseo_mlearning.api import sample_discipline

# %%
# Firstly,
# we define the discipline computing the Rosenbrock function
# and the Euclidean distance to the optimum:
from numpy import array

discipline = create_discipline(
    "AnalyticDiscipline",
    expressions={"z": "(1-x)**2+100*(y-x**2)**2", "c": "((x-1)**2+(y-1)**2)**0.5"},
    name="Rosenbrock",
)

# %%
# Then, we create the design space:
design_space = create_design_space()
design_space.add_variable("x", l_b=-2.0, u_b=2.0, value=0.0)
design_space.add_variable("y", l_b=-2.0, u_b=2.0, value=0.0)

# %%
# Then,
# we sample the discipline with an optimal LHS:
dataset = sample_discipline(discipline, design_space, ["z", "c"], "OT_OPT_LHS", 30)

# %%
# before creating a surrogate discipline:
surrogate_discipline = create_surrogate("RBFRegressor", dataset)

# %%
# and using it for prediction:
surrogate_discipline.execute({"x": array([1.0])})
surrogate_discipline.cache.last_entry

# %%
# This surrogate discipline can be used in a scenario.
# The underlying regression model can also be assessed,
# with the R2 measure for instance:
r2 = R2Measure(surrogate_discipline.regression_model, True)
r2.compute_learning_measure()
r2.compute_cross_validation_measure()

# %%
# or with the root mean squared error:
rmse = RMSEMeasure(surrogate_discipline.regression_model, True)
rmse.compute_learning_measure()
rmse.compute_cross_validation_measure()

with Path("my_surrogate.pkl").open("wb") as f:
    pickle.dump(surrogate_discipline, f)

discipline = import_discipline("my_surrogate.pkl")
discipline.execute({"x": array([1.0])})
discipline.get_output_data()
