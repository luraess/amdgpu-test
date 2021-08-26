using AMDGPU, Test

function vadd!(c, a, b)
    i = workitemIdx().x
    c[i] = a[i] + b[i]
    return
end

function test_amdgpu()
    # cpu version
    N     = 32
    a     = rand(Float64, N)
    b     = rand(Float64, N)
    c_cpu = a + b
    # AMD gpu version
    a_d   = ROCArray(a)
    b_d   = ROCArray(b)
    c_d   = similar(a_d)
    # run gpu kernel
    wait(@roc groupsize=N vadd!(c_d, a_d, b_d))
    # gather
    c     = Array(c_d)
    # test
    return @test isapprox(c, c_cpu)
end

test_amdgpu()
