r"""
Uncertainty propagation
=======================

In this example,
we will propagate uncertainties through a discipline $f:u,v\mapsto u+v$
"""

from gemseo import create_discipline
from gemseo import create_parameter_space
from gemseo.uncertainty import create_statistics
from gemseo_mlearning.api import sample_discipline
from matplotlib import pyplot as plt

# %%
# Firstly,
# we define a uncertain space with two normal random variables $u$ and $v$
# with mean -1 and +1 and unit standard deviation.
uncertain_space = create_parameter_space()
uncertain_space.add_random_variable("u", "OTNormalDistribution", mu=-1.0)
uncertain_space.add_random_variable("v", "OTNormalDistribution", mu=1.0)

# %%
# Then,
# we define the discipline from analytic formula:
discipline = create_discipline("AnalyticDiscipline", expressions={"w": "u+v"})

# %%
# Thirdly,
# we sample the discipline with a Monte Carlo algorithm:
dataset = sample_discipline(discipline, uncertain_space, ["w"], "OT_MONTE_CARLO", 1000)

# %%
# Lastly,
# we create an `EmpiricalStatistics` object to estimate statistics,
# such as mean and variance:
statistics = create_statistics(dataset)
mean = statistics.compute_mean()
variance = statistics.compute_variance()
names = ["u", "v", "w"]
for name in names:
    print(name, mean[name][0], variance[name][0])

# %%
# !!! note
#
#     The mean and standard deviation of the output are almost equal to 0 and 2,
#     which is the expected behavior
#     of the sum of two independent Gaussian random variables.

# %%
# We can also plot the histogram of the three random variables:
fig, axes = plt.subplots(1, 3)
for ax, name in zip(axes, names):
    ax.hist(dataset.get_view(variable_names=name))
    ax.set_title(name)
plt.show()
