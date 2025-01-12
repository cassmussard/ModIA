"""# The H2TurboFan discipline."""

from typing import Iterable, Mapping

from gemseo.core.discipline import MDODiscipline

from numpy import array, ndarray

from lh2pac.turbofan_h2_function import fct_turbofan_h2


class H2TurboFan(MDODiscipline):
    """Wrapper of the MARILib-based function `fct_turbofan_h2`.

    This discipline evaluates the function `fct_turbofan_h2`
    from values of `TECHNOLOGICAL_VARIABLES` and `DESIGN_VARIABLES`
    passed to the method `execute` as a dictionary of NumPy arrays.

    The discipline uses `DEFAULT_DESIGN_VALUES` for unspecified `DESIGN_VARIABLES`
    and `DEFAULT_TECHNOLOGICAL_VALUES` for unspecified `TECHNOLOGICAL_VARIABLES`.
    """

    TECHNOLOGICAL_VARIABLES = ["tgi", "tvi", "sfc", "mass", "drag"]
    DESIGN_VARIABLES = ["thrust", "bpr", "area", "aspect_ratio"]
    OBJECTIVE = "mtow"
    CONSTRAINTS = ["tofl", "vapp", "vz_mcl", "vz_mcr", "oei_path", "ttc", "far"]
    OBSERVABLES = ["fuel", "coc"]
    OUTPUT_VARIABLES = [OBJECTIVE] + CONSTRAINTS + OBSERVABLES
    DEFAULT_DESIGN_VALUES = {
        "thrust": array([125000.0]),
        "bpr": array([8.5]),
        "area": array([160.0]),
        "aspect_ratio": array([9.5]),
    }
    DEFAULT_TECHNOLOGICAL_VALUES = {
        "tgi": array([0.3]),
        "tvi": array([0.845]),
        "drag": array([1.0]),
        "sfc": array([1.0]),
        "mass": array([1.0]),
    }

    def __init__(self) -> None:
        super(H2TurboFan, self).__init__()

        # Define the input and output variables.
        self.input_grammar.update_from_names(
            self.DESIGN_VARIABLES + self.TECHNOLOGICAL_VARIABLES
        )
        self.output_grammar.update_from_names(self.OUTPUT_VARIABLES)

        # Define the default inputs.
        self.default_inputs.update(self.DEFAULT_DESIGN_VALUES)
        self.default_inputs.update(self.DEFAULT_TECHNOLOGICAL_VALUES)

    def _run(self):
        """Run the wrapped MARILib function `fct_turbofan_h2`.

        1. Retrieve the inputs passed to `execute` and store in `local_data`.
        2. Execute the MARILib-based function `fct_turbofan_h2`.
        3. Store the results in `local_data`.
        """
        design_data = self.__get_variables(self.local_data, self.DESIGN_VARIABLES)
        techno_data = self.__get_variables(
            self.local_data, self.TECHNOLOGICAL_VARIABLES
        )
        output_data = {
            name: array([value])
            for name, value in fct_turbofan_h2(techno_data, design_data, "eval").items()
        }
        self.local_data.update(output_data)

    @staticmethod
    def __get_variables(
        data: Mapping[str, ndarray], names: Iterable[str]
    ) -> dict[str, float]:
        """Return the values of the variable readable by :meth:`fct_turbofan_h2`.

        Args:
            data: The data to be converted.
            names: The names of the variables.

        Returns:
            The data readable by :meth:`fct_turbofan_h2`.
        """
        return {name: data[name][0] for name in names}
