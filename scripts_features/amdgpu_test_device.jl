using AMDGPU

@inbounds function memcopy_triad!(A, B, C, s, nx2, side, sig, waitval)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    
    AMDGPU.device_signal_wait(sig, waitval)
    
    if side == 1 && ix <= nx2
        A[ix,iy] = B[ix,iy] + s*C[ix,iy]
    elseif side == 2 && ix > nx2
        A[ix,iy] = B[ix,iy] + 2s*C[ix,iy]
    end
    return
end

@inbounds function copy2buf!(Buf, A, nx2, sig, waitval)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y

    AMDGPU.device_signal_wait(sig, waitval)

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

    A   = AMDGPU.zeros(nx,ny)
    B   =  AMDGPU.ones(nx,ny)
    C   =  AMDGPU.ones(nx,ny)
    Buf = AMDGPU.zeros(nx2,ny)

    s = 1.0

    AMDGPU.device!(1)
    println("Selecting device $(AMDGPU.device())")

    sig_1 = AMDGPU.HSASignal(0)
    sig_2 = AMDGPU.HSASignal(0)
    sig_3 = AMDGPU.HSASignal(0)

    ret1 = @roc groupsize=threads gridsize=grid memcopy_triad!(A, B, C, s, nx2, 1, sig_1, 1)
    println("ret 1")
    ret2 = @roc groupsize=threads gridsize=grid memcopy_triad!(A, B, C, s, nx2, 2, sig_2, 1)
    println("ret 2")
    ret3 = @roc groupsize=threads gridsize=grid copy2buf!(Buf, A, nx2, sig_3, 1)
    println("ret 3")
    AMDGPU.HSA.signal_store_release(sig_1.signal[], 1)
    AMDGPU.HSA.signal_store_release(sig_2.signal[], 1)
    wait(ret1)
    AMDGPU.HSA.signal_store_release(sig_3.signal[], 1)
    wait(ret2)
    wait(ret3)

    # @assert A[1:nx2,:] ≈ Buf
    println("Done")

    return Buf
end

main()
