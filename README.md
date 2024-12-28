# InverseMotionPlanning.jl

This repository provides a Julia-based framework for motion planning and goal-driven trajectory prediction. It includes tools for trajectory sampling, optimization, visualization, and inference over motion planning problems. The framework is built around a probabilistic programming approach, leveraging [Gen.jl](https://github.com/probcomp/Gen.jl) for generative modeling and inference.

The approach and methodology in this repository are described in:
- **[Bayesian Inverse Motion Planning for Online Goal Inference in Continuous Domains](https://ztangent.github.io/assets/pdf/2023-inverse-motion-planning.pdf)** (ICRA Workshop on Cognitive Modeling in Robot Learning for Adaptive Human-Robot Interactions, 2023)
- **[Monte Carlo Methods for Motion Planning and Goal Inference](https://dspace.mit.edu/handle/1721.1/153789)** (MIT EECS Masters Thesis, 2024)

## 🧩 Features
#### 1.	Trajectory Sampling
Generate optimized motion trajectories in 2D and 3D environments using Monte Carlo sampling methods, including Random-Walk Metropolis-Hastings (RWMH), Metropolis-Adjusted Langevin Ascent (MALA), Hamiltonian Monte Carlo (HMC), Newtonian Monte Carlo (NMC), and hybrids
  
#### 2. Inverse Motion Planning
Compute posterior distributions over trajectories and goal regions using particle filters and Bayesian techniques.
    
#### 3. Scene Design and Visualization
Create and visualize complex 2D/3D environments with obstacles, regions of interest, and safety constraints.
      
#### 4. Goal Inference
Infer probabilistic goal distributions given partial trajectory observations.
      
#### 5. Heuristic Baselines
Implement simple cost-based baselines for predicting goal probabilities, including the closest goal heuristic and a Laplace approximation method.

## 📂 Repository Structure

The repository is organized as follows:
```
InverseMotionPlanning.jl/
├── examples/                
│   ├── baselines.jl         # Heuristic baselines for goal prediction
│   ├── goal_trajectory_2D.jl # 2D goal trajectory generation and goal inference
│   ├── scene_design_2D.jl   # Designing 2D scenes with obstacles and regions
│   ├── trajectory_2D.jl     # 2D trajectory sampling and optimization
│   ├── trajectory_3D.jl     # 3D trajectory sampling and optimization
├── src/                     
│   ├── analysis_utils.jl    # Utility functions for analyzing trajectories
│   ├── callbacks.jl         # Callback functions for logging and visualization
│   ├── costs.jl             # Cost functions for trajectory evaluation
│   ├── distributions.jl     # Distributions for goal and trajectory inference
│   ├── gen_utils.jl         # General utilities for sampling and modeling
│   ├── geometry.jl          # Geometric primitives and calculations
│   ├── inference.jl         # Inference functions (e.g., particle filtering)
│   ├── InverseMotionPlanning.jl # Main package module
│   ├── scenes.jl            # Scene creation and manipulation
│   ├── smc_trajectory_gf.jl # Sequential Monte Carlo for trajectory sampling
│   ├── trajectory_gf.jl     # Trajectory generative functions
├── Project.toml             # Package dependencies
├── README.md                
```

## 🔧 Getting Started

### Cloning the repository
```
git clone https://github.com/InverseMotionPlanning/InverseMotionPlanning.jl
cd InverseMotionPlanning.jl
```

### Installation & Setup
Ensure you have [Julia](https://julialang.org/) installed (v1.7 or later is recommended). In the cloned repository, enter `Pkg` mode in the Julia REPL by pressing <kbd>]</kbd>, then run:

```
instantiate
```

This installs all required dependencies from `Project.toml`, including:
- [Gen.jl](https://github.com/probcomp/Gen.jl) (probabilistic programming)
- [GLMakie.jl](https://github.com/MakieOrg/Makie.jl) (visualization)
- [Meshes.jl](https://github.com/JuliaGeometry/Meshes.jl) (geometry manipulation).

Now exit `Pkg` mode by pressing <kbd>←</kbd> or <kbd>Ctrl-C</kbd>, and run:

```julia
using InverseMotionPlanning
```

to call functions in the package.

### Usage

After running `using InverseMotionPlanning`, you can now run the examples in the `examples` directory:
- `trajectory_2D.jl` constructs an example 2D scene and samples trajectories in that scene
- `trajectory_3D.jl` constructs an example 3D scene and samples trajectories in that scene

 

### Modifying and Extending the Framework: 
- To add a custom cost function, modify `costs.jl` and integrate it into the inference pipeline.
- To change inference strategies or MCMC kernels, edit `inference.jl` or implement your own kernel.
- For new scenes or geometric primitives, extend `scenes.jl` or `geometry.jl`.


## 🤝 Contributing
Contributions are welcome! To get started:
1. Fork this repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request describing your changes.

## 📖 Citing this Work
If you find this code useful in your research, please cite the following:

```
@inproceedings{zhixuan2023bayesian,
  title = {Bayesian Inverse Motion Planning for Online Goal Inference in Continuous Domains},
  author = {Zhi-Xuan, Tan and Kondic, Jovana and Slocum, Stewart and Tenenbaum, Joshua B and Mansinghka, Vikash K and Hadfield-Menell, Dylan},
  booktitle = {ICRA 2023 Workshop on Cognitive Modeling in Robot Learning},
  year = {2023},
  month = jun,
  url = {https://sites.google.com/view/cognitive-modeling-icra2023-ws/contributions?authuser=0#h.dk14d3kbwe65},
}
```

```
@mastersthesis{kondic2024monte,
  title = {Monte Carlo Methods for Motion Planning and Goal Inference},
  author = {Kondic, Jovana},
  year = {2024},
  school = {Massachusetts Institute of Technology},
  url = {https://hdl.handle.net/1721.1/153789}
}
```

## 📄 License
This project is licensed under the Apache License. See the LICENSE.md file for details.
