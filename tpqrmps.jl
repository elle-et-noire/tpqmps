using ITensors
using LinearAlgebra
using Plots

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

function transverseIsing()
  N = 16 # number of site
  χ = 40 # virtual bond dimension of Ψ
  χeff = χ # truncate dimension
  d = 2 # physical dimension
  J = 1
  g = 1
  l = 1 # larger than maximum energy density
  kmax = 1 # num to cool down
  Ψbonds = siteinds(χ, N + 1)
  physinds = siteinds(d, N)

  Ψ = [ITensor(leftunitary(χ, d), physinds[i], Ψbonds[i], Ψbonds[i + 1]) for i in 1:N]

  hbonds = siteinds(3, N - 1)
  leftmold = zeros(d, 3, d)
  rightmold = zeros(d, 3, d)
  leftmold[:, 1, :] = rightmold[:, 3, :] = I(2)
  leftmold[:, 2, :] = rightmold[:, 2, :] = sqrt(J) * [1 0;0 -1]
  leftmold[:, 3, :] = rightmold[:, 1, :] = -g * [0 1;1 0]
  rightmold2 = deepcopy(rightmold)
  rightmold2[:, 1, :] -= l * I(2) # I ⊗ ⋯ ⊗ (-l*I+σx)
  h_left = ITensor(leftmold, physinds[1], hbonds[1], physinds[1]')
  h_right = ITensor(rightmold / N, physinds[end], hbonds[end], physinds[end]') # coef "N" smashed in rH
  l_h_right = ITensor(-rightmold2 / N, physinds[end], hbonds[end], physinds[end]') # coef "-N" smashed in rH2
  middlemold = zeros(3, d, 3, d)
  middlemold[1, :, 1, :] = I(2)
  middlemold[3, :, 3, :] = I(2)
  middlemold[1, :, 2, :] = middlemold[2, :, 3, :] = sqrt(J) * [1 0;0 -1]
  middlemold[1, :, 3, :] = -g * [0 1;1 0]
  # l - h
  l_h = [ITensor(middlemold, hbonds[i-1], physinds[i], hbonds[i], physinds[i]') for i in 2:N-1]
  # h : hamiltonian density
  # hamdens = [ITensor(middlemold, hbonds[i-1], physinds[i], hbonds[i], physinds[i]') for i in 2:N-1]
  pushfirst!(l_h, h_left)
  # pushfirst!(hamdens, h_left)
  push!(l_h, l_h_right)
  hamdens = deepcopy(l_h)
  # push!(hamdens, h_right)
  hamdens[end] = h_right

  singleh = reduce(*, hamdens)
  D, _ = eigen(singleh, physinds, physinds')
  D |> storage |> display

  # h2 = lH * h[2] * δ(mpobonds[2], mpobonds[N-1]) * rH
  # a, b, c, e, m, n = inds(h2)
  # C = combiner(a, c, m)
  # C2 = combiner(b, e, n)
  # display(matrix(C * h2 * C2))

  # ket_k_can = Vector{ITensor}(undef, N)
  # ket_k_canbonds = Vector{{Index{Int64}}}(undef, N + 1)
  ket_k_can = deepcopy(Ψ)
  ket_k_canbonds = deepcopy(Ψbonds)
  # ket_k_canbonds[1] = Ψbonds[1]
  # ket_k_canbonds[end] = Ψbonds[end]
  enes = []
  for k in 1:kmax
    #======== left-canonical ========#
    U, λ, V = svd(Ψ[1] * l_h[1] |> noprime, (Ψbonds[1]))
    # ignore U @ |aux> since nothing operate here and contraction is same as unit matrix.
    ket_k_can[1] = replaceind(λ, commonind(U, λ) => Ψbonds[1]) * V

    for site in 1:N-1
      U, λ, V = svd((ket_k_can[site] * Ψ[site+1]) * l_h[site+1] |> noprime, (ket_k_canbonds[site], physinds[site]))
      ket_k_canbonds[site+1] = commonind(U, λ)
      ket_k_can[site] = U
      ket_k_can[site+1] = λ * V
    end

    #======== right-canonical ========#
    U, λ, V = svd(ket_k_can[end], (ket_k_canbonds[N+1]))
    # ignore V @ |aux> since nothing operate here and contraction is same as unit matrix.
    ket_k_can[end] = V * replaceind(λ, commonind(λ, U) => ket_k_canbonds[end])

    for site in N-1:-1:1
      U, λ, V = svd(ket_k_can[site] * ket_k_can[site+1], (ket_k_canbonds[site+2], physinds[site+1]))
      ket_k_canbonds[site+1] = Index(χeff)
      ket_k_can[site+1] = δ(ket_k_canbonds[site+1], commonind(λ, U)) * U # truncate
      ket_k_can[site] = V * λ * δ(ket_k_canbonds[site+1], commonind(λ, U))
    end

    Ψnorm2 = prime(ket_k_can[1], ket_k_canbonds[2]) * conj(ket_k_can[1])
    for site in 2:N-1
      Ψnorm2 *= prime(ket_k_can[site], ket_k_canbonds[site], ket_k_canbonds[site+1])
      Ψnorm2 *= conj(ket_k_can[site])
    end
    Ψnorm2 *= prime(ket_k_can[end], ket_k_canbonds[end-1])
    Ψnorm2 *= conj(ket_k_can[end])
    println("k=$k, Ψnorm2 = ", Ψnorm2[])
    Ψnorm2 = real(Ψnorm2[])

    # energy density
    uk = prime(ket_k_can[1], ket_k_canbonds[2], physinds[1]) * hamdens[1] * conj(ket_k_can[1])
    for site in 2:N-1
      uk *= prime(ket_k_can[site], ket_k_canbonds[site], ket_k_canbonds[site+1], physinds[site])
      uk *= conj(ket_k_can[site])
      uk *= hamdens[site]
    end
    uk *= prime(ket_k_can[end], ket_k_canbonds[end-1], physinds[end])
    uk *= conj(ket_k_can[end])
    uk *= hamdens[end]
    println(real(uk[]) / Ψnorm2)

    push!(enes, real(uk[]) / Ψnorm2)

    Ψ = deepcopy(ket_k_can / Ψnorm2^inv(2N))
    Ψbonds = deepcopy(ket_k_canbonds)
  end

  println("==== end ====")
  display(enes)
  println("\n==== time record ====\n")
  kBT = [N * (l - enes[k]) / (2k) for k in 1:kmax]

  plot(kBT, enes)
  savefig("energy.png")
  # println(Ψ * conj(Ψ))


  # Ψnorm2 = prime(Ψ[1], Ψbonds[2]) * conj(Ψ[1])
  # for site in 2:N-1
  #   Ψnorm2 *= prime(Ψ[site], Ψbonds[site], Ψbonds[site+1]) * conj(Ψ[site])
  # end
  # Ψnorm2 *= prime(Ψ[N], Ψbonds[N]) * conj(Ψ[N])
end

ITensors.set_warn_order(33)
@time transverseIsing()
