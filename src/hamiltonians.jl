# ISING 1D MPO
#------------------------------------------------------------------------------------------------------
function Ising_1D(N::Int64, J::Float64, h::Float64, sites::Vector{IT.Index{Int64}}; is_periodic::Bool=true)

    os = ITM.OpSum()
    
    for j=1:N-1
        os += 4*J,"Sz",j,"Sz",j+1
    end

    if is_periodic == true
        os += 4*J,"Sz",1,"Sz",N
    end

    for j=1:N
        os += -2*h,"Sx",j
    end

    return ITM.MPO(os,sites)
    
end

# ISING 2D MPO
#------------------------------------------------------------------------------------------------------
function Ising_2D(Nx::Int64, Ny::Int64, J::Float64, h::Float64, sites::Vector{IT.Index{Int64}}; is_periodic::Bool=true)
    
    os = ITM.OpSum()
    
    for x=1:Nx
        for y=1:Ny-1
            j1 = (x-1)*Ny + y       
            j2 = j1 + 1             
            os += 4*J, "Sz", j1, "Sz", j2
        end
    end

    for x=1:Nx-1
        for y=1:Ny
            j1 = (x-1)*Ny + y       
            j2 = j1 + Ny            
            os += 4*J, "Sz", j1, "Sz", j2
        end
    end

    if is_periodic
        for x=1:Nx
            j1 = (x-1)*Ny + 1
            j2 = x*Ny
            os += 4*J, "Sz", j1, "Sz", j2
        end

        for y=1:Ny
            j1 = y
            j2 = (Nx-1)*Ny + y
            os += 4*J, "Sz", j1, "Sz", j2
        end
    end

    for x=1:Nx
        for y=1:Ny
            j = (x-1)*Ny + y
            os += -2*h, "Sx", j
        end
    end

    return ITM.MPO(os, sites)
    
end
