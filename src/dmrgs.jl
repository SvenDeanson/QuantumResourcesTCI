# Compute groundstate of 1D Ising using ITensorMPS
#------------------------------------------------------------------------------------------------------
function run_Ising_1D_DMRG(N::Int64, J::Float64, h::Float64, verbose::Int64, periodic::Bool)

    nsweeps = 20
    maxdim = [5,10,50]
    cutoff = [1E-6]

    sites = IT.siteinds("S=1/2", N)  
    psi0 = ITM.random_mps(sites)
    # psi0 is initialized from the analytical expression near classical point h=0 to help convergence
    if J == 1.0
        psi0 = W_states(sites)
    else
        psi0 = GHZ_states(sites)
    end

    Id = pauli_string_mpo(sites, fill(1, N))
    PrX = pauli_string_mpo(sites, fill(2, N))
    Pr = (Id + PrX)
    Pr[1] = 1/2 * Pr[1]

    H = Ising_1D(N, J, h, sites, is_periodic = periodic)

    energy,psi = ITM.dmrg(H,psi0;nsweeps,maxdim,cutoff, noise=1e-2, outputlevel=verbose);

    new_psi = Pr * psi
    IT.normalize!(new_psi)
    final_psi = new_psi
    
    return final_psi, energy, check_max_bond(final_psi)
end

#------------------------------------------------------------------------------------------------------
function run_Ising_2D_DMRG(Nx::Int64, Ny::Int64, J::Float64, h::Float64, verbose::Int64, periodic::Bool)

    nsweeps = 20
    maxdim = [5, 10, 50, 100]
    cutoff = [1E-8]

    L = Nx * Ny

    sites = IT.siteinds("S=1/2", L)  
    psi0 = ITM.random_mps(sites)

    Id = pauli_string_mpo(sites, fill(1, L))
    PrX = pauli_string_mpo(sites, fill(2, L))
    Pr = (Id + PrX)
    Pr[1] = 1/2 * Pr[1]

    H = Ising_2D(Nx, Ny, J, h, sites, is_periodic = periodic)

    energy,psi = ITM.dmrg(H,psi0;nsweeps,maxdim,cutoff, noise=1e-2, outputlevel=verbose);

    new_psi = Pr * psi
    IT.normalize!(new_psi)
    final_psi = new_psi
    
    return final_psi, energy, check_max_bond(final_psi)
end