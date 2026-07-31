<div align="center">

<h1 style="font-size: 3rem; margin-bottom: 0.25rem;">
  Homotopy-Conditioned MPPI
</h1>

<h2 style="font-size: 1.5rem; font-weight: 400; margin-top: 0;">
  Bridging Geometric Planning and Dynamics-Aware Control
</h2>

<br>

<h3>
  <a href="#overview">Overview</a>
  &nbsp;·&nbsp;
  <a href="#animations">Animations</a>
  &nbsp;·&nbsp;
  <a href="#methods">Methods</a>
  &nbsp;·&nbsp;
  <a href="#experiments">Experiments</a>
  &nbsp;·&nbsp;
  <a href="#usage">Usage</a>
</h3>

</div>

<p align="center">
  <img src="figures/trajectory_prior_comparison.png" alt="Priors used to inititialize the MPPI controller: on the left, the empirical trajectories, on the right the extracted gaussian representation" width="100%">
</p>

<p align="center">
  <em>Priors used to inititialize the MPPI controller: on the left, the empirical trajectories, on the right the extracted gaussian representation.</em>
</p>

<p align="center">
  <img src="figures/ackermann_all_variants.png" alt="Controller trajectories across no-wall, static-wall, and dynamic two-wall scenarios with every MPPI variant" width="100%">
</p>

<p align="center">
  <em>Controller trajectories across the no-wall, static-blockage, and dynamic two-wall experimental conditions with every MPPI variant (priors).</em>
</p>

---

## Overview

Model Predictive Path Integral (MPPI) control is a sampling-based, receding-horizon method for nonlinear systems. Its performance depends heavily on the proposal distribution: proposals that are too narrow can miss feasible routes, while overly broad proposals waste a limited rollout budget.

This project improves MPPI by conditioning its proposals on geometric trajectories grouped by homotopy class. A topology-aware fish-school planner discovers collision-free routes around obstacles, and the resulting trajectory priors guide MPPI toward distinct topological alternatives.

The geometric prior changes **where MPPI samples**. It does not replace the system model or modify the shared rollout objective. Dynamic feasibility is still evaluated through the predictive dynamics at every control step.

### Key ideas

<table>
  <tr>
    <td width="50%" valign="top">
      <strong>1. Discover diverse routes</strong><br>
      Generate collision-free paths using a topology-aware stochastic planner.
    </td>
    <td width="50%" valign="top">
      <strong>2. Separate homotopy classes</strong><br>
      Group trajectories by homotopy signature.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <strong>3. Build trajectory priors</strong><br>
      Represent each route class with empirical paths, mean trajectories, and covariance information.
    </td>
    <td width="50%" valign="top">
      <strong>4. Convert geometry into controls</strong><br>
      Localize each path and convert it into model-specific nominal controls.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <strong>5. Condition MPPI sampling</strong><br>
      Allocate rollouts around multiple route-conditioned proposals.
    </td>
    <td width="50%" valign="top">
      <strong>6. Preserve fair comparisons</strong><br>
      Evaluate every controller with the same dynamics and objective.
    </td>
  </tr>
</table>

# Animations
The animations below compare **Standard MPPI** with **Gaussian-prior MPPI** under the same system and scenario.

### Unicycle

#### No wall

| Standard MPPI | Gaussian-prior MPPI |
|:---:|:---:|
| ![Unicycle standard MPPI, no wall](gifs/unicycle/no_wall/standard_mppi/animation.gif) | ![Unicycle Gaussian-prior MPPI, no wall](gifs/unicycle/no_wall/gaussian_prior_mppi/animation.gif) |

#### Static blockage

| Standard MPPI | Gaussian-prior MPPI |
|:---:|:---:|
| ![Unicycle standard MPPI, static two-wall scenario](gifs/unicycle/static/standard_mppi/animation.gif) | ![Unicycle Gaussian-prior MPPI, static two-wall scenario](gifs/unicycle/static/gaussian_prior_mppi/animation.gif) |

#### Dynamic blockage

| Standard MPPI | Gaussian-prior MPPI |
|:---:|:---:|
| ![Unicycle standard MPPI, dynamic two-wall scenario](gifs/unicycle/dynamic/standard_mppi/animation.gif) | ![Unicycle Gaussian-prior MPPI, dynamic two-wall scenario](gifs/unicycle/dynamic/gaussian_prior_mppi/animation.gif) |

### Ackermann vehicle

#### No wall

| Standard MPPI | Gaussian-prior MPPI |
|:---:|:---:|
| ![Ackermann standard MPPI, no wall](gifs/ackerman/no_wall/standard_mppi/animation.gif) | ![Ackermann Gaussian-prior MPPI, no wall](gifs/ackerman/no_wall/gaussian_prior_mppi/animation.gif) |

#### Static blockage — two walls

| Standard MPPI | Gaussian-prior MPPI |
|:---:|:---:|
| ![Ackermann standard MPPI, static two-wall scenario](gifs/ackerman/static/standard_mppi/animation.gif) | ![Ackermann Gaussian-prior MPPI, static two-wall scenario](gifs/ackerman/static/gaussian_prior_mppi/animation.gif) |

#### Dynamic blockage — two walls

| Standard MPPI | Gaussian-prior MPPI |
|:---:|:---:|
| ![Ackermann standard MPPI, dynamic two-wall scenario](gifs/ackerman/dynamic/standard_mppi/animation.gif) | ![Ackermann Gaussian-prior MPPI, dynamic two-wall scenario](gifs/ackerman/dynamic/gaussian_prior_mppi/animation.gif) |

All variants use the same task objective. The comparison therefore isolates the effect of proposal construction, covariance representation, rollout allocation, and explicit mode selection.

# Methods
### Supported systems

The same homotopy-conditioned proposal mechanism is evaluated across two vehicle models with different levels of dynamic complexity.

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>Unicycle</h3>
      <p>
        A compact model used for controlled ablations and rapid experimentation.
      </p>
      <p>
        <strong>State</strong><br>
        Planar position and heading
      </p>
      <p>
        <strong>Control</strong><br>
        Translational velocity and angular velocity
      </p>
      <p>
        The unicycle model isolates the effect of the proposal distribution while keeping the system dynamics simple and interpretable.
      </p>
    </td>
    <td width="50%" valign="top">
      <h3>Ackermann</h3>
      <p>
        A bicycle model used to test whether the same proposal strategy transfers to richer vehicle dynamics.
      </p>
      <p>
        <strong>State</strong><br>
        Planar position, heading, speed, and steering angle
      </p>
      <p>
        <strong>Control</strong><br>
        Acceleration and steering-rate commands
      </p>
      <p>
        This model introduces nonholonomic steering behavior and additional dynamic constraints.
      </p>
    </td>
  </tr>
</table>

> The interactive viewer supports both systems in a single application. The active vehicle model and controller variant can be changed directly from the interface.

---

# Experiments

Each controller is evaluated under three environmental conditions.

| Condition | Prior generation | Execution environment | Purpose |
|:--|:--|:--|:--|
| **No wall** | Original environment | Original environment | Measures nominal navigation performance in the base cluttered scene. |
| **Static blockage** | Additional walls are present | Same blocked environment | Tests planning and control when the obstruction is known in advance. |
| **Dynamic blockage** | Additional walls are absent | Walls are introduced during execution | Tests adaptation when a previously feasible route becomes blocked. |

# Usage
## Repository organization

```text
├── geometry/                 Geometric and collision-checking utilities
├── graph/                    Graph construction and topology components
├── planner/                  Fish-school trajectory generation
├── runs_unicycle.py          Unicycle dynamics and controller variants
├── runs_ackerman.py          Ackermann dynamics and controller variants
├── viewer.py                 Interactive visualization interface
├── requirements.txt          Project dependencies
```

## Getting started

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it according to your operating system.

### 2. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Launch the interactive viewer

```bash
python viewer.py
```

The viewer provides a unified interface for both supported vehicle models. From the control panel you can:

- Switch between the **Unicycle** and **Ackermann** systems.
- Select any available controller variant.
- Run all experimental scenarios.
- Inspect sampled trajectories, covariance ellipses, obstacle predictions, and control rollouts.

Changing the active system automatically reloads the corresponding simulation module and updates the visualization.

### 4. Reproduce the experiments

To regenerate all experiment results, run the simulation scripts for each vehicle model.

```bash
# Unicycle experiments
python runs_unicycle.py

# Ackermann experiments
python runs_ackerman.py
```

Each script evaluates every controller variant across all experimental conditions and saves the resulting trajectories, controls, covariance estimates, and execution statistics.

---

