# Design a liquid hydrogen powered aircraft

The aim of this project is to 
design a liquid hydrogen powered aircraft (LH2PAC)
by combining a numerical simulator 
with three mathematical fields,
namely surrogate modeling, uncertainty quantification and optimization:

- optimization algorithms are commonly used to solve design problems,
- uncertainty quantification (UQ) is becoming increasingly popular 
  to take into account various sources of uncertainties
  that may occur during the design cycle,
- surrogate models are enablers for optimization and UQ
  because the numerical simulators are often computationally expensive
  and do not allow a sufficient number of evaluations to be obtained 
  to estimate statistics properly or to converge a numerical optimization algorithm.

## Problems

### Introduction

The design problem aims to minimize the maximal take-off weight of an aircraft
whilst ensuring some constraints such as a maximal take-off field length
and a maximal approach speed.

For this purpose,
it will be possible to play with four design parameters.

By the way,
the design takes place in an uncertain environment
where the technological choices can be probabilized.

In the following,
we will note $f:x,u\mapsto f(x,u)$
the MARILib-based model of a liquid hydrogen powered model
where $x$ are the design parameters and $u$ the uncertain ones.

We will assume that its computational cost is too high 
and we will try to replace it with a surrogate model
to address the following problem. 

### Problem 1 - Surrogate modeling and optimization

We will create a surrogate model of $g:x\mapsto g(x)=f(x,u_{\mathrm{default}})$
to approximate the objective and constraints of the design problem
with respect to the design parameters $x$.
Then,
we will use this surrogate model in an optimization process
to minimize the objective whilst ensuring the constraints
by varying the design parameters.

### Problem 2 - Surrogate modeling and uncertainty quantification

We will create a surrogate model of $h:u\mapsto h(x)=f(x_{\mathrm{default}},u)$
to approximate the objective and constraints of the design problem
with respect to the uncertain parameters $u$.
Then,
we will use this surrogate model in an uncertainty study
to propagate the uncertainty related to the technological choices,
quantify the resulting output uncertainty
and explain it from the uncertainty sources (sensitivity analysis).


### Problem 3 - Surrogate modeling and robust optimization

We will create a surrogate model of $h:x,u\mapsto h(x)=f(x,u)$
to approximate the objective and constraints of the design problem
with respect to both design parameters $x$ and uncertain parameters $u$.
Then,
we will use this surrogate model to to seek the best aircraft concept
taking into account the uncertainty of technological choices.

## Software

To solve these problems,
we will combine
the [MARILib](https://github.com/marilib/MARILib_obj>) Python library 
for aircraft modeling,
and the [GEMSEO](https://gitlab.com/gemseo/dev/gemseo>) Python library 
for problem definition, optimization, surrogate modeling and uncertainty quantification.

GEMSEO relies on popular Python libraries,
such as 
[scikit-learn](https://github.com/scikit-learn/scikit-learn>) for surrogate modeling,
[OpenTURNS](https://github.com/openturns/openturns>) for uncertainty quantification,
[NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for scientific computing
and [matplotlib](https://matplotlib.org/) for visualization.
