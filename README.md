# InverseMotionPlanning.jl

Inverse motion planning in Julia. 

## Setup

Enter `Pkg` mode in the Julia REPL by pressing `]`, then run:

```
instantiate
```

to install all required dependencies. Now exit `Pkg` mode and run:

```julia
using InverseMotionPlanning
```

to call functions in the package.

## Usage

After running `using InverseMotionPlanning`, you can now run the examples in the `examples` direction:
- `trajectory_2D.jl` constructs an example 2D scene and samples trajectories in that scene
- `trajectory_3D.jl` constructs an example 3D scene and samples trajectories in that scene