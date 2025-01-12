"""
# Create a design space

A `DesignSpace` defines the space in which the design variables belongs
and is required to set the optimization problem,
in the same way as the objective and the constraints.

## Create a design space

The simplest is to use the function `create_design_space`:
"""

from gemseo import create_design_space

design_space = create_design_space()

# %%
# This design space can include a design variable $x$
# without bounds and without current value:
design_space.add_variable("x")

# %%
# a design variable $y$ of dimension 2
# with a lower bound and a current value:
from numpy import array

design_space.add_variable("y", size=2, l_b=0.0, value=array([0.5, 0.75]))

# %%
# as well as a design variable $z$
# with both lower and upper bounds but without default value:
design_space.add_variable("z", l_b=-1.0, u_b=1.0)

# %%
# Let's take a look at this design space:
design_space

# %%
# ## Create a class of design space
# If we want to use this design space more than once,
# it can be more convenient and Pythonic to use the object-oriented paradigm
# and subclass `DesignSpace`:

from gemseo.algos.design_space import DesignSpace


class MyDesignSpace(DesignSpace):
    def __init__(self):
        super().__init__(name="foo")
        self.add_variable("x")
        self.add_variable("y", size=2, l_b=0.0, value=array([0.5, 0.75]))
        self.add_variable("z", l_b=-1.0, u_b=1.0)


# %%
# Then,
# we only have to instantiate `MyDesignSpace`:
design_space = MyDesignSpace()
design_space
