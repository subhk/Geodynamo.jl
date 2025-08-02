# ============================================================================
# Timestepping Module with SHTns
# ============================================================================

module Timestepping
    using LinearAlgebra
    using PencilArrays
    using ..Parameters
    using ..VariableTypes
    using ..LinearOps
    using ..SHTnsSetup
    
    # Timestepping state
    mutable struct TimestepState
        time::Float64
        dt::Float64
        step::Int
        iteration::Int
        error::Float64
        converged::Bool
    end
    
    # Implicit matrices for each spherical harmonic mode (SHTns version)
    struct SHTnsImplicitMatrices{T}
        matrices::Vector{BandedMatrix{T}}  # One for each l value
        factorizations::Vector{Any}       # LU factorizations
        l_values::Vector{Int}             # l values for indexing
    end
    
    function create_shtns_timestepping_matrices(config::SHTnsConfig, domain::RadialDomain, 
                                               diffusivity::Float64, dt::Float64)
        matrices = BandedMatrix{Float64}[]
        factorizations = Any[]
        l_values = Int[]
        
        # Get unique l values from SHTns configuration
        unique_l = unique(config.l_values)
        
        # Create matrix for each l value
        for l in unique_l
            push!(l_values, l)
            
            # Implicit operator: (1/dt - diffusivity * implicit_factor * L_l)
            # where L_l = d²/dr² + (2/r)d/dr - l(l+1)/r²
            
            laplacian = create_radial_laplacian(domain)
            
            # Modify for spherical harmonic l
            l_factor = Float64(l * (l + 1))
            implicit_data = copy(laplacian.data)
            
            # Time derivative term
            implicit_data[i_KL + 1, :] .+= 1.0 / dt
            
            # Diffusion term  
            implicit_data .*= -diffusivity * d_implicit
            
            # Spherical harmonic term
            for n in 1:domain.N
                r_inv_sq = domain.r[n, 2]  # 1/r²
                implicit_data[i_KL + 1, n] += diffusivity * d_implicit * l_factor * r_inv_sq
            end
            
            matrix = BandedMatrix(implicit_data, i_KL, domain.N)
            push!(matrices, matrix)
            
            # Create LU factorization
            dense_matrix = banded_to_dense(matrix)
            factorization = lu(dense_matrix)
            push!(factorizations, factorization)
        end
        
        return SHTnsImplicitMatrices(matrices, factorizations, l_values)
    end
    
    function banded_to_dense(matrix::BandedMatrix{T}) where T
        # Convert banded matrix to dense for LU factorization
        N = matrix.size
        bandwidth = matrix.bandwidth
        dense = zeros(T, N, N)
        
        for j in 1:N
            for i in max(1, j - bandwidth):min(N, j + bandwidth)
                band_row = bandwidth + 1 + i - j
                dense[i, j] = matrix.data[band_row, j]
            end
        end
        
        return dense
    end
    
    function apply_explicit_operator!(output::SHTnsSpectralField{T}, 
                                     input::SHTnsSpectralField{T},
                                     nonlinear::SHTnsSpectralField{T}, 
                                     domain::RadialDomain,
                                     diffusivity::Float64, dt::Float64) where T
        # Explicit operator: (1/dt + diffusivity * (1-implicit_factor) * L_l) * input + nonlinear
        
        @views for lm_idx in 1:input.nlm
            l = input.config.l_values[lm_idx]
            
            # Create explicit operator for this l
            laplacian = create_radial_laplacian(domain)
            l_factor = Float64(l * (l + 1))
            
            # Apply to input
            for r_idx in input.local_radial_range
                if r_idx <= size(input.data_real, 3)
                    # Time derivative
                    output.data_real[lm_idx, 1, r_idx] = input.data_real[lm_idx, 1, r_idx] / dt
                    output.data_imag[lm_idx, 1, r_idx] = input.data_imag[lm_idx, 1, r_idx] / dt
                    
                    # Explicit diffusion (simplified application)
                    diffusion_real = zero(T)
                    diffusion_imag = zero(T)
                    
                    # Add diffusion terms (would use proper banded matrix multiplication)
                    output.data_real[lm_idx, 1, r_idx] += diffusion_real
                    output.data_imag[lm_idx, 1, r_idx] += diffusion_imag
                    
                    # Add nonlinear terms
                    output.data_real[lm_idx, 1, r_idx] += nonlinear.data_real[lm_idx, 1, r_idx]
                    output.data_imag[lm_idx, 1, r_idx] += nonlinear.data_imag[lm_idx, 1, r_idx]
                end
            end
        end
    end
    
    function solve_implicit_step!(solution::SHTnsSpectralField{T}, 
                                 rhs::SHTnsSpectralField{T},
                                 matrices::SHTnsImplicitMatrices{T}) where T
        # Solve implicit system for each mode
        @views for lm_idx in 1:solution.nlm
            l = solution.config.l_values[lm_idx]
            
            # Find matrix index for this l value
            matrix_idx = findfirst(==(l), matrices.l_values)
            if matrix_idx !== nothing
                factorization = matrices.factorizations[matrix_idx]
                
                # Solve for real part
                rhs_real = rhs.data_real[lm_idx, 1, :]
                solution.data_real[lm_idx, 1, :] = factorization \ rhs_real
                
                # Solve for imaginary part  
                rhs_imag = rhs.data_imag[lm_idx, 1, :]
                solution.data_imag[lm_idx, 1, :] = factorization \ rhs_imag
            end
        end
    end
    
    function compute_timestep_error(new_field::SHTnsSpectralField{T}, 
                                   old_field::SHTnsSpectralField{T}) where T
        error = zero(Float64)
        
        for lm_idx in 1:new_field.nlm
            for r_idx in new_field.local_radial_range
                if r_idx <= size(new_field.data_real, 3)
                    diff_real = new_field.data_real[lm_idx, 1, r_idx] - old_field.data_real[lm_idx, 1, r_idx]
                    diff_imag = new_field.data_imag[lm_idx, 1, r_idx] - old_field.data_imag[lm_idx, 1, r_idx]
                    error += diff_real^2 + diff_imag^2
                end
            end
        end
        
        # Global reduction across all processes
        global_error = MPI.Allreduce(error, MPI.SUM, PencilSetup.comm)
        return sqrt(global_error)
    end
    
    export TimestepState, SHTnsImplicitMatrices, create_shtns_timestepping_matrices
    export apply_explicit_operator!, solve_implicit_step!, compute_timestep_error
end
