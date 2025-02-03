# CHECK MAXIMUM BOND DIMENSION
#------------------------------------------------------------------------------------------------------
function check_max_bond(psi)

    res = 0
    for tt in psi
        idx_ll = IT.inds(tt)
        for idx in idx_ll
            if IT.hastags(idx, "S=1/2,Site") == false
                res = idx.space > res ? idx.space : res
            end
        end
    end
    return res

end

# INITIAL 1D GROUND STATES HELPERS
#------------------------------------------------------------------------------------------------------

function W_states(sites::Vector{IT.Index{Int64}})
    
    N = length(sites)
    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    state_ll = [state]
    
    for i=1:N-1
        push!(state_ll, circshift(state, i))
    end
    
    state = [isodd(n) ? "Dn" : "Up" for n in 1:N]
    push!(state_ll, state)
    
    for i=1:N-1
        push!(state_ll, circshift(state, i))
    end
    
    gs = 0.0 .* ITM.random_mps(sites)
    
    for state in state_ll
        gs = gs + ITM.MPS(sites, state)
    end
    
    gs = IT.normalize!(gs)

    return gs

end

function GHZ_states(sites::Vector{IT.Index{Int64}})
    
    N = length(sites)
    stateUp = ["Up" for n in 1:N]
    stateDn = ["Dn" for n in 1:N]
    state_ll = [stateUp, stateDn]

    gs = 0.0 .* ITM.random_mps(sites)
    
    for state in state_ll
        gs = gs + ITM.MPS(sites, state)
    end
    
    gs = IT.normalize!(gs)

    return gs

end

function Neel_states(sites::Vector{IT.Index{Int64}})

    N = length(sites)
    state1 = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    state_ll = [state1]
    state2 = [isodd(n) ? "Dn" : "Up" for n in 1:N]
    
    push!(state_ll, state2)
    
    gs = 0.0 .* IT.random_mps(sites)
    
    for state in state_ll
        gs = gs + IT.MPS(sites, state)
    end
    
    gs = IT.normalize!(gs)

    return gs

end

# MAXIMUM BOND DIMENSION CHECK
#------------------------------------------------------------------------------------------------------
function check_max_bond(psi::ITM.MPS)

    res = 0

    for tt in psi
        
        idx_ll = IT.inds(tt)
        
        for idx in idx_ll
            if IT.hastags(idx, "S=1/2,Site") == false
                res = idx.space > res ? idx.space : res
            end
        end

    end

    return res

end


# PAULI MATRICES
#------------------------------------------------------------------------------------------------------

ID = [1.0 0.0; 0.0 1.0]
SX = [0.0 1.0; 1.0 0.0]
SY = [0.0 -1.0im; 1.0im 0.0]
SZ = [1.0 0.0; 0.0 -1.0]

ALPHA = [ID, SX, SY, SZ]

function pauli_string_mpo(sites::Vector{IT.Index{Int64}}, str::Vector{Int64})

    N = length(sites)
    Op = ITM.MPO(sites)
    for i in 1:N

        op_matrix = ALPHA[str[i]]

        pauli_mat = IT.ITensor(op_matrix, sites[i]', sites[i])

        Op[i] = pauli_mat
    end
    return Op
end
