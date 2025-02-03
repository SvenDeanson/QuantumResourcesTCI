# Get component from the state
#------------------------------------------------------------------------------------------------------
function extract_value(psi::ITM.MPS, sites::Vector{IT.Index{Int64}}, str::Vector{Int64})

    V = IT.ITensor(1.)
    for j=1:length(psi)
      V *= (psi[j]*IT.state(sites[j],str[j]))
    end
    
    return IT.scalar(V)
end


# Computing scalar entropy
#------------------------------------------------------------------------------------------------------
function get_scalar_entropy(psi::ITM.MPS, sites::Vector{IT.Index{Int64}}, str::Vector{Int64})
    coeff = extract_value(psi, sites, str)
    return -(coeff^2) * log2(coeff^2) 
end

# Computation of Quantum Coherence using TensorCrossInterpolation
#------------------------------------------------------------------------------------------------------
function get_quantum_coherence(psi::ITM.MPS; tolerance::Float64=1e-8, maxdim::Int64=200, output::Int64=0, tci_path::Union{String, Nothing} = nothing)

    sites = IT.siteinds(psi)
    N = length(psi)
    f(str) = get_scalar_entropy(psi, sites, str)
    localdims = fill(2, N)
    cf = TCI.CachedFunction{Float64}(f, localdims)
    tci, ranks, errors = TCI.crossinterpolate2(Float64, cf, localdims; tolerance=tolerance, maxbonddim=maxdim, verbosity=output)

    if tci_path != nothing
        tt = TCI.TensorTrain(tci)
        mps = TCC.MPS(tt)
        
        f = h5open(tci_path, "w")
        write(f, "psi", mps)
        close(f)
    end 
    
    return sum(tci), tci, ranks, errors
end
