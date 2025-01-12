from typing import Iterable
from typing import Mapping

from gemseo.core.scenario import Scenario
from gemseo.algos.opt_problem import OptimizationProblem
from numpy import array
from numpy import atleast_1d
from numpy import ndarray

from lh2pac.gemseo.discipline import H2TurboFan
from lh2pac.marilib.utils import unit
from lh2pac.turbofan_h2_function import fct_turbofan_h2
from lh2pac.turbofan_h2_function import str_h2turbofan


def _get_variables(
    data: Mapping[str, ndarray], names: Iterable[str]
) -> dict[str, float]:
    """Return the values of the variable readable by :meth:`fct_turbofan_h2`.

    Args:
        data: The data to be converted.
        names: The names of the variables.

    Returns:
        The data readable by :meth:`fct_turbofan_h2`.
    """
    _data = {}
    for name in names:
        value = data[name]
        if isinstance(value, ndarray):
            _data[name] = value[0]
        else:
            _data[name] = value
    return _data


_default_inputs = {
    "thrust": array([125000.0]),
    "bpr": array([8.5]),
    "area": array([160.0]),
    "aspect_ratio": array([9.5]),
    "tgi": array([0.3]),
    "tvi": array([0.845]),
    "drag": array([1.0]),
    "sfc": array([1.0]),
    "mass": array([1.0]),
}


def _get_data(obj):
    if isinstance(obj, Mapping):
        _obj = _default_inputs.copy()
        _obj.update(obj)
        return _obj

    if isinstance(obj, H2TurboFan):
        return obj.local_data

    if isinstance(obj, Scenario):
        obj = obj.formulation.opt_problem

    if not isinstance(obj, OptimizationProblem):
        obj = OptimizationProblem.from_hdf(obj)

    return _get_optimum(obj)


def draw_aircraft(obj=None, title=""):
    data = _get_data({} if obj is None else obj)
    design_data = _get_variables(data, H2TurboFan.DESIGN_VARIABLES)
    techno_data = _get_variables(data, H2TurboFan.TECHNOLOGICAL_VARIABLES)
    fct_turbofan_h2(techno_data, design_data, "draw", title=title)


def get_aircraft_data(obj):
    data = _get_data(obj)
    design_data = _get_variables(data, H2TurboFan.DESIGN_VARIABLES)
    techno_data = _get_variables(data, H2TurboFan.TECHNOLOGICAL_VARIABLES)
    output_data = _get_variables(data, H2TurboFan.OUTPUT_VARIABLES)
    return str_h2turbofan(techno_data, design_data, output_data)


def _get_optimum(problem):
    if not isinstance(problem, OptimizationProblem):
        problem = OptimizationProblem.from_hdf(problem)

    f_opt, x_opt, is_feas, c_opt, _ = problem.get_optimum()
    y_opt = problem.database[x_opt]
    x_opt = {
        name: x_opt[i] for i, name in enumerate(problem.design_space.variable_names)
    }
    data = {name: array([value]) for name, value in x_opt.items()}
    data.update({k: atleast_1d(v) for k, v in y_opt.items()})
    data.update(H2TurboFan.DEFAULT_TECHNOLOGICAL_VALUES)
    data["ttc"] += unit.s_min(25)
    data["tofl"] += 2200
    data["vapp"] += unit.mps_kt(137)
    data["vz_mcr"] *= -1
    data["vz_mcl"] = -data["vz_mcl"] + unit.mps_ftpmin(300)
    data["oei_path"] = -data["oei_path"] + 0.011
    data["far"] += 13.4
    return data
