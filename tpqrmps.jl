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

function cooling(Ψk, Ψprev, Ψbonds, Dχbonds, l_h)

end

function energydensity()
  N = 16 # number of site
  χ = 40 # virtual bond dimension of Ψ
  d = 2 # physical dimension
  l = 4 # larger than maximum energy density
  kmax = 200 # num to cool down
  Ψbonds = siteinds(χ, N + 1)
  physinds = siteinds(d, N)

  Ψprev = [ITensor(leftunitary(χ, d), physinds[i], Ψbonds[i], Ψbonds[i + 1]) for i in 1:N]

  hamdens, l_h = hamdens_transverseising(N, physinds, l)

  Ψk = Vector{ITensor}(undef, N)
  Dχbonds = Vector{Index}(undef, N + 1)
  Dχbonds[1] = Ψbonds[1]
  Dχbonds[end] = Ψbonds[end]
  uks = Vector{Float64}(undef, kmax)
  for k in 1:kmax
    #======== left-canonical ========#
    U, λ, V = svd(Ψprev[1] * l_h[1] |> noprime, (Ψbonds[1]))
    # ignore U @ |aux> since nothing operate here and U works same as unit matrix.
    Ψk[1] = replaceind(λ, commonind(U, λ) => Ψbonds[1]) * V

    for site in 1:N-1
      U, λ, V = svd((Ψk[site] * Ψprev[site+1]) * l_h[site+1] |> noprime, (Dχbonds[site], physinds[site]))
      Dχbonds[site+1] = commonind(U, λ)
      Ψk[site] = U
      Ψk[site+1] = λ * V
    end

    #======== right-canonical ========#
    U, λ, V = svd(Ψk[end], (Ψbonds[end]))
    # ignore V @ |aux> since nothing operate here and contraction is same as unit matrix.
    Ψk[end] = V * replaceind(λ, commonind(λ, U) => Ψbonds[end])

    for site in N-1:-1:1
      U, λ, V = svd(Ψk[site] * Ψk[site+1], (Ψbonds[site+2], physinds[site+1]))
      Ψk[site+1] = δ(Ψbonds[site+1], commonind(λ, U)) * U # truncate
      Ψk[site] = V * λ * δ(Ψbonds[site+1], commonind(λ, U))
    end

    #======== Ψnorm2 = <k|k> ========#
    Ψnorm2 = prime(Ψk[1], Ψbonds[2]) * conj(Ψk[1])
    for site in 2:N-1
      Ψnorm2 *= prime(Ψk[site], Ψbonds[site], Ψbonds[site+1])
      Ψnorm2 *= conj(Ψk[site])
    end
    Ψnorm2 *= prime(Ψk[end], Ψbonds[end-1])
    Ψnorm2 *= conj(Ψk[end])
    println("k=$k, Ψnorm2 = ", Ψnorm2[])
    Ψnorm2 = real(Ψnorm2[])

    #======== uₖ = Eₖ/N ========#
    uₖ= prime(Ψk[1], Ψbonds[2], physinds[1]) * hamdens[1] * conj(Ψk[1])
    for site in 2:N-1
      uₖ *= prime(Ψk[site])
      uₖ *= conj(Ψk[site])
      uₖ *= hamdens[site]
    end
    uₖ *= prime(Ψk[end], Ψbonds[end-1], physinds[end])
    uₖ *= conj(Ψk[end])
    uₖ *= hamdens[end]
    uks[k] = real(uₖ[]) / Ψnorm2
    println("uk = ", uks[k])

    Ψprev = deepcopy(Ψk / Ψnorm2^inv(2N))
  end

  println("==== end ====")
  display(uks)
  kBT = [N * (l - uks[k]) / 2k for k in 1:kmax]
  open("energy-density-l=$l.txt", "w") do fp
    content = ""
    for (t, uk) in zip(kBT, uks)
      content *= "$t $uk\n"
    end
    write(fp, content)
  end
  plot(kBT, uks)
  savefig("energy-density.png")
  println("\n\n==== time record ====")
end

@time energydensity()
