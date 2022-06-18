# https://juliaphysics.github.io/PhysicsTutorials.jl/tutorials/general/quantum_ising/quantum_ising.html

using LinearAlgebra

⊗(x, y) = kron(x, y)

function main()
  σᶻ = [1 0; 0 -1]
  σˣ = [0 1; 1 0]
  σᶻ⊗σᶻ |> display
end

function TransverseFieldIsing(;N,h)
  id = [1 0; 0 1]
  σˣ = [0 1; 1 0]
  σᶻ = [1 0; 0 -1]

  # vector of operators: [σᶻ, σᶻ, id, ...]
  first_term_ops = fill(id, N)
  first_term_ops[1] = σᶻ
  first_term_ops[2] = σᶻ

  # vector of operators: [σˣ, id, ...]
  second_term_ops = fill(id, N)
  second_term_ops[1] = σˣ

  H = zeros(Int, 2^N, 2^N)
  for i in 1:N-1
      # tensor multiply all operators
      H -= foldl(⊗, first_term_ops)
      # cyclic shift the operators
      first_term_ops = circshift(first_term_ops,1)
  end

  for i in 1:N
      H -= h*foldl(⊗, second_term_ops)
      second_term_ops = circshift(second_term_ops,1)
  end
  H
end

"""
Binary `BitArray` representation of the given integer `num`, padded to length `N`.
"""
bit_rep(num::Integer, N::Integer) = BitArray(parse(Bool, i) for i in string(num, base=2, pad=N))

"""
    generate_basis(N::Integer) -> basis

Generates a basis (`Vector{BitArray}`) spanning the Hilbert space of `N` spins.
"""
function generate_basis(N::Integer)
    nstates = 2^N
    basis = Vector{BitArray{1}}(undef, nstates)
    for i in 0:nstates-1
        basis[i+1] = bit_rep(i, N)
    end
    return basis
end

function TransverseFieldIsing_explicit(; N::Integer, h::T=0) where T<:Real
  basis = generate_basis(N)
  H = zeros(T, 2^N, 2^N)
  bonds = zip(collect(1:N-1), collect(2:N))
  for (i, bstate) in enumerate(basis)
      # diagonal part
      diag_term = 0.
      for (site_i, site_j) in bonds
          if bstate[site_i] == bstate[site_j]
              diag_term -= 1
          else
              diag_term += 1
          end
      end
      H[i, i] = diag_term

      # off diagonal part
      for site in 1:N
          new_bstate = copy(bstate)
          # flip the bit on the site (that's what σˣ does)
          new_bstate[site] = !new_bstate[site]
          # find corresponding single basis state with unity overlap (orthonormality)
          new_i = findfirst(isequal(new_bstate), basis)
          H[i, new_i] = -h
      end
  end
  return H
end