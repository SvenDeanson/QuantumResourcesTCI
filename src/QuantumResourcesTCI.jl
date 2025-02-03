module QuantumResourcesTCI

# Import external dependencies
import ITensors as IT
import ITensorMPS as ITM
import TensorCrossInterpolation as TCI
import TCIITensorConversion as TCC

using CSV
using HDF5
using DataFrames
using ProgressMeter

# Include all submodules/files
include("utilities.jl")
include("dmrgs.jl")
include("hamiltonians.jl")
include("quantum_coherence.jl")
include("quantum_magic.jl")

# Re-export functions (if needed)
# export some_function_from_utilities, another_function_from_dmrgs

export Ising_1D, Ising_2D
export run_Ising_1D_DMRG, run_Ising_2D_DMRG

export get_scalar_entropy, get_quantum_coherence
export pauli_string_expectation, get_scalar_magic, get_quantum_magic

end  # module QuantumResourcesTCI
