"""# Unit conversion

The data presented in the use case description are often expressed with usual units
but the simulator requires standard units.
This example illustrates how to convert a data from a unit to another.

First,
we load the `unit` module.
"""

from lh2pac.marilib.utils import unit

# %%
# Then,
# we consider a time value expressed in minutes:
time_in_minutes = 1

# %%
# Lastly,
# we convert it into seconds:
time_in_seconds = unit.s_min(time_in_minutes)
time_in_seconds
