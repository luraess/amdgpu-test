# Feature tests

Playground to develop and test features required to support [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) as GPU backend in [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) and [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) (later).

## Async kernel execution
AMDGPU supports queues, the analogy of CUDA's streams.

Visual profiling can be achieved using `rocprof` tool and inspecting the `hsa-trace` (here on Satori):

```
/opt/rocm/bin/rocprof --hsa-trace ~/julia_x86/julia-1.7.2/bin/julia amdgpu_test.jl
```

The `.json` output file can be viewed using the Chrome browser opening `chrome://tracing` in Chrome:

![](/docs/rocprof_chrome_tracer.png)
