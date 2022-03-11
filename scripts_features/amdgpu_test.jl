using AMDGPU

@inbounds function memcopy_triad!(A, B, C, s, nx2, side)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    if side == 1 && ix <= nx2
        A[ix,iy] = B[ix,iy] + s*C[ix,iy]
    elseif side == 2 && ix > nx2
        A[ix,iy] = B[ix,iy] + 2s*C[ix,iy]
    end
    return
end

@inbounds function copy2buf!(Buf, A, nx2)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    if ix <= nx2
        Buf[ix,iy] = A[ix,iy]
    end
    return
end


function main()
    println("AMDGPU functional: $(AMDGPU.functional())")

    nx, ny  = 16*1024, 16*1024
    threads = (32, 8)
    grid    = (nx, ny)
    nx2     = Int(round(nx/2))
    nrep    = 5

    A   = AMDGPU.zeros(nx,ny)
    B   =  AMDGPU.ones(nx,ny)
    C   =  AMDGPU.ones(nx,ny)
    Buf = AMDGPU.zeros(nx2,ny)

    s = 1.0

    AMDGPU.device!(1) # Replace AMDGPU.set_default_agent!(AMDGPU.get_agents(:gpu)[1])
    println("Selecting device $(AMDGPU.device())")

    qs = Vector{AMDGPU.HSAQueue}(undef,2)
    for iside = 1:2
        qs[iside] = AMDGPU.HSAQueue(get_default_agent())
        priority = iside == 1 ? AMDGPU.HSA.AMD_QUEUE_PRIORITY_HIGH : AMDGPU.HSA.AMD_QUEUE_PRIORITY_LOW
        AMDGPU.HSA.amd_queue_set_priority(qs[iside].queue,priority)
    end

    signals = Vector{AMDGPU.RuntimeEvent{AMDGPU.HSAStatusSignal}}(undef,2)

    for irep = 1:nrep
        for iside = 1:2
            signals[iside] = @roc groupsize=threads gridsize=grid queue=qs[iside] memcopy_triad!(A, B, C, s, nx2, iside)
            @async wait(signals[iside])
        end

        # for iside = 1:2 # the same as @async in the loop above
        #     wait(signals[iside])
        # end

        for iside = 1
            signals[iside] = @roc groupsize=threads gridsize=grid queue=qs[iside] copy2buf!(Buf, A, nx2)
            wait(signals[iside]) # without @async here host blocks until cpy2buf is done before going to next irep
            # one could use @async if next task is totally independent 
        end
    end
    @assert A[1:nx2,:] ≈ Buf

    println("Done")

    return Buf
end

main()
