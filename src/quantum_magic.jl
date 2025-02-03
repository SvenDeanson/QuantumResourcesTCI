# Precomputing Pauli operators that will be used for computation saves time
#------------------------------------------------------------------------------------------------------
function precompute_pauli_ops(sites::Vector{IT.Index{Int64}})
    
    n_sites = length(sites)
    ops_array = Array{IT.ITensor}(undef, 4, n_sites)

    for n in 1:n_sites
        ops_array[1, n] = IT.op("Id", sites[n])
        ops_array[2, n] = 2.0 * IT.op("Sx", sites[n])
        ops_array[3, n] = 2.0 * IT.op("Sy", sites[n])
        ops_array[4, n] = 2.0 * IT.op("Sz", sites[n])
    end

    return ops_array
end

# Optimal contraction scheme for inner products of MPS
#------------------------------------------------------------------------------------------------------
function efficient_inner_product(psi::ITM.MPS, phi::ITM.MPS)
    
    E = phi[1]' * psi[1]
    for n in 2:length(psi)
        E = E * phi[n]' * psi[n]
    end

    return IT.scalar(E)
end

# Apply Pauli operators to transform psi to phi and preform inner product
#------------------------------------------------------------------------------------------------------
function pauli_string_expectation(psi::ITM.MPS, phi::ITM.MPS, ops_array::Array{IT.ITensor}, pauli_string::Vector{Int})

    for (n, pauli_index) in enumerate(pauli_string)
        local_op = ops_array[pauli_index, n]  
        phi[n] = local_op * psi[n]
    end

    expectation_value = efficient_inner_product(phi, psi)

    return expectation_value
end

# f(x)=|x|^4
#------------------------------------------------------------------------------------------------------
function get_scalar_magic(psi::ITM.MPS, phi::ITM.MPS, ops_array::Array{IT.ITensor}, pauli_string::Vector{Int})
    
    coeff = pauli_string_expectation(psi, phi, ops_array, pauli_string)

    return abs(coeff)^4
end

# Computation of Stabilizer Rényi-2 Entropy using TensorCrossInterpolation
#------------------------------------------------------------------------------------------------------
function get_quantum_magic(psi::ITM.MPS; tolerance::Float64=1e-6, maxdim::Int64=100, output::Int64=0, tci_path::Union{String, Nothing} = nothing)
    
    N = length(psi)
    sites = IT.siteinds(psi)
    ops_array = precompute_pauli_ops(sites)
    phi = ITM.MPS(sites)

    func(str) = get_scalar_magic(psi, phi, ops_array, str)
    func(fill(1,N))

    localdims = fill(4, N)
    # Use BigInt for caching
    cf = TCI.CachedFunction{Float64, BigInt}(func, localdims)

    result, elapsed_time, gc_time, allocations = @timed begin
        tci, ranks, errors = TCI.crossinterpolate2(Float64, cf, localdims, [fill(1, length(localdims)),fill(2, length(localdims)),fill(3, length(localdims)),fill(4, length(localdims))];  
        tolerance=tolerance, maxbonddim=maxdim, verbosity=output)
    end

    if tci_path != nothing
        tt = TCI.TensorTrain(tci)
        mps = TCC.MPS(tt)
        
        f = h5open(tci_path, "w")
        write(f, "psi", mps)
        close(f)
    end 
    
    return N-log2(sum(tci)), tci, ranks, errors
    
end
