# InverseTAMP.jl

Inverse task-and-motion-planning (TAMP) in Julia. 

(The "task" part isn't implemented yet.)

## Setup

Enter `Pkg` mode in the Julia REPL by pressing `]`, then run:

```
instantiate
```

to install all required dependencies. Now exit `Pkg` mode and run:

```julia
using InverseTAMP
```

to call functions in the package.

## Usage

After running `using InverseTAMP`, you can now run the examples in the `examples` direction:
- `trajectory_2D.jl` constructs an example 2D scene and samples trajectories in that scene
- `trajectory_3D.jl` constructs an example 3D scene and samples trajectories in that scene