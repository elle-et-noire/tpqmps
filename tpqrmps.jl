using ITensors
using LinearAlgebra
using Plots
using Printf

function boxmuller()
  x = rand()
  y = rand()
  return sqrt(-2 * log(x)) * exp(2pi * im * y)
end

function leftunitary(χ, d)
  q, _ = reshape([boxmuller() for i in 1:(χ*d)^2], (χ * d, χ * d)) |> qr
  u = reshape(q, (d, χ, d, χ))
  return u[:, :, 1, :] # u[1, :, :, :] is right unitary
end

function hamdens_transverseising(sitenum, physinds, l)
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
  hamdens = deepcopy(l_h)
  hamdens[end] = h_right

  return (hamdens, l_h)
end

function cooldown_seqtrunc(Ψk, Ψprev, Ψbonds, Dχbonds, physinds, l_h)
  N = length(Ψk)
  #======== left-canonical ========#
  U, S, V = svd(Ψprev[1] * l_h[1] |> noprime, (Ψbonds[1]))
  # ignore U @ |aux> since nothing operate here and U works same as unit matrix.
  Ψk[1] = replaceind(S, commonind(U, S) => Ψbonds[1]) * V

  for site in 1:N-1
    U, S, V = svd((Ψk[site] * Ψprev[site+1]) * l_h[site+1] |> noprime, (Dχbonds[site], physinds[site]))
    Dχbonds[site+1] = commonind(U, S)
    Ψk[site] = U
    Ψk[site+1] = S * V
  end

  #======== right-canonical ========#
  U, S, V = svd(Ψk[end], (Ψbonds[end]))
  # ignore V @ |aux> since nothing operate here and contraction is same as unit matrix.
  Ψk[end] = V * replaceind(S, commonind(S, U) => Ψbonds[end])

  for site in N-1:-1:1
    U, S, V = svd(Ψk[site] * Ψk[site+1], (Ψbonds[site+2], physinds[site+1]))
    Ψk[site+1] = δ(Ψbonds[site+1], commonind(S, U)) * U # truncate
    Ψk[site] = V * S * δ(Ψbonds[site+1], commonind(S, U))
  end
end

function norm2(Ψ, Ψbonds)
  N = length(Ψ)
  Ψnorm2 = prime(Ψ[1], Ψbonds[2]) * conj(Ψ[1])
  for site in 2:N-1
    Ψnorm2 *= prime(Ψ[site], Ψbonds[site], Ψbonds[site+1])
    Ψnorm2 *= conj(Ψ[site])
  end
  Ψnorm2 *= prime(Ψ[end], Ψbonds[end-1])
  Ψnorm2 *= conj(Ψ[end])
  return real(Ψnorm2[])
end

function expectedvalue(Ψ, Ψbonds, physinds, mpo, Ψnorm2)
  N = length(Ψ)
  val= prime(Ψ[1], Ψbonds[2], physinds[1]) * mpo[1] * conj(Ψ[1])
  for site in 2:N-1
    val *= prime(Ψ[site])
    val *= conj(Ψ[site])
    val *= mpo[site]
  end
  val *= prime(Ψ[end], Ψbonds[end-1], physinds[end])
  val *= conj(Ψ[end])
  val *= mpo[end]
  return real(val[]) / Ψnorm2
end

function entanglemententropy(Ψ, Ψbonds, physinds, subrange)
  U, S, V = svd(foldl(*, Ψ), physinds[subrange])
  return -sum(storage(S) .|> x -> x^2 * log(x^2))
end

function energydensity(;l)
  N = 16 # number of site
  χ = 40 # virtual bond dimension of Ψ
  d = 2 # physical dimension
  kmax = 200 # num to cool down
  Ψbonds = siteinds(χ, N + 1)
  physinds = siteinds(d, N)
  Ψprev = [ITensor(leftunitary(χ, d), physinds[i], Ψbonds[i], Ψbonds[i + 1]) for i in 1:N]
  hamdens, l_h = hamdens_transverseising(N, physinds, l)

  Ψ = [Ψprev, Vector{ITensor}(undef, N)]
  Dχbonds = Vector{Index}(undef, N + 1)
  Dχbonds[1] = Ψbonds[1]
  Dχbonds[end] = Ψbonds[end]
  uₖs = Vector{Float64}(undef, kmax)

  for k in 1:kmax
    cooldown_seqtrunc(Ψ[k&1+1], Ψ[2-k&1], Ψbonds, Dχbonds, physinds, l_h)
    #======== Ψnorm2 = <k|k> ========#
    Ψnorm2 = norm2(Ψ[k&1+1], Ψbonds)
    println("k=$k, Ψnorm2 = $Ψnorm2")
    #======== uₖ = Eₖ/N ========#
    uₖs[k] = expectedvalue(Ψ[k&1+1], Ψbonds, physinds, hamdens, Ψnorm2)
    println("uk = ", uₖs[k])
    Ψ[k&1+1] /= Ψnorm2^inv(2N)
  end

  println("==== end ====")
  kBT = [N * (l - uₖs[k]) / 2k for k in 1:kmax]
  open("energy-density-l=$l.txt", "w") do fp
    content = ""
    for (t, uk) in zip(kBT, uₖs)
      content *= "$t $uk\n"
    end
    write(fp, content)
  end
  plot(kBT, uₖs)
  savefig("energy-density.png")
  println("\n\n==== time record ====")
end

@time energydensity(l=64)
