using AMDGPU,Statistics,Printf

function main()
    gpu_agent = first(AMDGPU.get_agents(:gpu))
    cpu_agent = first(AMDGPU.get_agents(:cpu))

    AMDGPU.set_default_agent!(gpu_agent)

    A_h = rand(16*1024,16*1024)
    A   = AMDGPU.zeros(Float64,size(A_h))

    A_h_ptr    = pointer(A_h)
    locked_ref = Ref{Ptr{Cvoid}}()
    agents     = [cpu_agent.agent,gpu_agent.agent]
    AMDGPU.HSA.amd_memory_lock(Ptr{Cvoid}(A_h_ptr), sizeof(A_h), agents, 2, locked_ref) |> AMDGPU.check
    A_h_lptr   = locked_ref[]

    ntrial = 20; nskip = 2; el_sec = Float64[]
    for it = 1:(nskip+ntrial)
        println(it)
        # tel = @elapsed AMDGPU.HSA.memory_copy(pointer(A),A_lptr,sizeof(A)) |> AMDGPU.check
        # tel = @elapsed AMDGPU.HSA.memory_copy(A_lptr,pointer(A),sizeof(A)) |> AMDGPU.check
        tel = @elapsed AMDGPU.HSA.memory_copy(pointer(A),pointer(A_h),sizeof(A)) |> AMDGPU.check
        if it>nskip push!(el_sec,tel) else continue end
        # tel = @elapsed AMDGPU.HSA.memory_copy(pointer(A_h),pointer(A),sizeof(A)) |> AMDGPU.check
    end
    GC.gc()


    AMDGPU.HSA.amd_memory_unlock(A_h_lptr) |> AMDGPU.check

    GBs = sizeof(A)/1e9/minimum(el_sec)

    println("Memory bandwith: $GBs GB/s")

    return
end

main()