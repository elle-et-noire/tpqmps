using ITensors
using LinearAlgebra
using Plots
using Printf
using Random
using Base.Threads
using BenchmarkTools
@show nthreads()


function genΨgauss(;sitenum, physdim, bonddim, withaux=true)
  rng = MersenneTwister(1234)
  Ψbonds = siteinds(bonddim, sitenum + 1)
  physinds = siteinds(physdim, sitenum)

  Ψ = [ITensor(randn(rng, ComplexF64, (physdim, bonddim, bonddim)),
        physinds[site], Ψbonds[site], Ψbonds[site+1]) for site in 1:sitenum]
  if !withaux
    Ψbonds[1], Ψbonds[end] = Index(1), Index(1)
    Ψ[1] = ITensor(randn(rng, ComplexF64, (physdim, bonddim)), physinds[1], Ψbonds[1], Ψbonds[2])
    Ψ[end] = ITensor(randn(rng, ComplexF64, (physdim, bonddim)), physinds[end], Ψbonds[end-1], Ψbonds[end])
  end
  return Ψ, Ψbonds, physinds
end

function hdens_TIsing(sitenum, physinds, l)
  J = 1; g = 1
  d = dim(physinds[1])

  hbonds = siteinds(3, sitenum - 1)
  leftmold = zeros(d, 3, d)
  rightmold = zeros(d, 3, d)
  leftmold[:, 1, :] = rightmold[:, 3, :] = I(2)
  leftmold[:, 2, :] = rightmold[:, 2, :] = sqrt(J) * [1 0;0 -1]
  leftmold[:, 3, :] = rightmold[:, 1, :] = -g * [0 1;1 0]
  rightmold2 = deepcopy(rightmold)
  rightmold2[:, 1, :] -= l * I(2) # I ⊗ ⋯ ⊗ (-l*I+σx)
  h_left = ITensor(leftmold, physinds[1], hbonds[1], physinds[1]')
  h_right = ITensor(rightmold / sitenum, physinds[end], hbonds[end], physinds[end]') # coef "N" smashed in rH
  l_h_right = ITensor(-rightmold2 / sitenum, physinds[end], hbonds[end], physinds[end]') # coef "-N" smashed in rH2
  middlemold = zeros(3, d, 3, d)
  middlemold[1, :, 1, :] = I(2)
  middlemold[3, :, 3, :] = I(2)
  middlemold[1, :, 2, :] = middlemold[2, :, 3, :] = sqrt(J) * [1 0;0 -1]
  middlemold[1, :, 3, :] = -g * [0 1;1 0]
  # l - h
  l_h = [ITensor(middlemold, hbonds[i-1], physinds[i], hbonds[i], physinds[i]') for i in 2:sitenum-1]
  pushfirst!(l_h, h_left)
  push!(l_h, l_h_right)
  # h : hamiltonian density
  hdens = deepcopy(l_h)
  hdens[end] = h_right

  return (hdens, l_h)
end

"λ : array of singular values(not necessarily normalized)"
function sings2ent(λ)
  nrm = λ.^2 |> sum |> sqrt
  return -sum(λ ./ nrm .|> x -> 2x^2*log(x))
end

function cooldown_seqtrunc(Ψcur, Ψprev, Ψbonds, Dχbonds, physinds, l_h; measents=false, entropies=[])
  N = length(Ψcur)
  #======== left-canonical ========#
  # U, S, V = svd(Ψprev[1] * l_h[1] |> noprime, (Ψbonds[1]))
  # ignore U @ |aux> since nothing operate here and U works same as unit matrix.
  # Ψcur[1] = replaceind(S, commonind(U, S) => Ψbonds[1]) * V
  Ψcur[1] = Ψprev[1]

  for site in 1:N-1
    U, S, V = svd((Ψcur[site] * Ψprev[site+1]) * l_h[site+1] |> noprime, (Dχbonds[site], physinds[site]))
    Dχbonds[site+1] = commonind(U, S)
    Ψcur[site] = U
    Ψcur[site+1] = S * V
  end

  #======== right-canonical ========#
  U, S, V = svd(Ψcur[end], (Ψbonds[end]))
  # ignore U @ |aux> since nothing operate here and contraction is same as unit matrix.
  Ψcur[end] = V * replaceind(S, commonind(S, U) => Ψbonds[end])
  if measents
    entropies[end] = sings2ent(storage(S))
  end

  for site in N-1:-1:1
    U, S, V = svd(Ψcur[site] * Ψcur[site+1], (Ψbonds[site+2], physinds[site+1]))
    Ψcur[site+1] = δ(Ψbonds[site+1], commonind(S, U)) * U # truncate
    Ψcur[site] = V * S * δ(Ψbonds[site+1], commonind(S, U))
    if measents
      entropies[site+1] = sings2ent(storage(S))
    end
  end
  if measents
    _, S, _ = svd(Ψcur[1], Ψbonds[1])
    entropies[1] = sings2ent(storage(S))
  end
end

function norm2(Ψ, Ψbonds; diag=true, Ψ2=Ψ)
  N = length(Ψ)
  ret = prime(Ψ[1], Ψbonds[2]) * conj(Ψ2[1])
  for site in 2:N-1
    ret *= prime(Ψ[site], Ψbonds[site], Ψbonds[site+1])
    ret *= conj(Ψ2[site])
  end
  ret *= prime(Ψ[end], Ψbonds[end-1])
  ret *= conj(Ψ2[end])
  if diag
    return real(ret[])
  end
  return ret[]
end

function withaux(;N, χ, d, l, rep)
  Ψprev, Ψbonds, physinds = genΨgauss(sitenum=N, bonddim=χ, physdim=d, withaux=true)
  norm2₀ = norm2(Ψprev, Ψbonds)
  Ψprev /= norm2₀^inv(2N)
  hdens, l_h = hdens_TIsing(N, physinds, l)

  Ψ = [Ψprev, Vector{ITensor}(undef, N)]
  Dχbonds = Vector{Index}(undef, N + 1)
  Dχbonds[1] = Ψbonds[1]
  Dχbonds[end] = Ψbonds[end]

  for k in 1:rep
    cooldown_seqtrunc(Ψ[k&1+1], Ψ[2-k&1], Ψbonds, Dχbonds, physinds, l_h)
  end
  return norm2(Ψ[1], Ψbonds)
end

function withoutaux(;N, χ, d, l, rep)
  norms = zeros(Float64, χ^2)
  for num in 1:χ^2
    Ψprev, Ψbonds, physinds = genΨgauss(sitenum=N, bonddim=χ, physdim=d, withaux=false)
    norm2₀ = norm2(Ψprev, Ψbonds)
    Ψprev /= norm2₀^inv(2N)
    hdens, l_h = hdens_TIsing(N, physinds, l)

    Ψ = [Ψprev, Vector{ITensor}(undef, N)]
    Dχbonds = Vector{Index}(undef, N + 1)
    Dχbonds[1] = Ψbonds[1]
    Dχbonds[end] = Ψbonds[end]

    for k in 1:rep
      cooldown_seqtrunc(Ψ[k&1+1], Ψ[2-k&1], Ψbonds, Dχbonds, physinds, l_h)
    end
    norms[num] = norm2(Ψ[1], Ψbonds)
  end
  return sum(norms)
end

a = withaux(N=32, χ=40, d=2, l=5, rep=3)
println(a)
# b = withoutaux(N=32, χ=40, d=2, l=5, rep=3)
# println(b)
@btime withaux(N=32, χ=40, d=2, l=5, rep=3)
# @btime withoutaux(N=32, χ=40, d=2, l=5, rep=3)
