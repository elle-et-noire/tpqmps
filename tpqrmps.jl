using ITensors
using LinearAlgebra
using Plots
using Printf

function boxmuller()
  x = rand()
  y = rand()
  return sqrt(-2 * log(x)) * exp(2pi * im * y)
end

function unitary3ord(physind; leftbond, rightbond, rightunitary=false)
  χ = maximum(dim(leftbond), dim(rightbond))
  d = dim(physind)
  q, _ = reshape([boxmuller() for i in 1:(χ*d)^2], (χ * d, χ * d)) |> qr
  u = reshape(q, (d, χ, d, χ))
  if !rightunitary
    return ITensor(u[:, 1:dim(leftbond), 1, 1:dim(rightbond)], physind, leftbond, rightbond) # leftunitary
  end
  return ITensor(u[1, 1:dim(leftbond), :, 1:dim(rightbond)], leftbond, physind, rightbond) # rightunitary
end

function genΨcan(;sitenum, physdim, bonddim, withaux=true, rightunitary=false)
  Ψbonds = siteinds(bonddim, sitenum + 1)
  physinds = siteinds(physdim, sitenum)
  Ψ = [unitary3ord(physinds[i], leftbond=Ψbonds[i], rightbond=Ψbonds[i + 1], rightunitary=rightunitary) for i in 1:sitenum]
  if !withaux
    Ψbonds[1], Ψbonds[end] = Index(1), Index(1)
    Ψ[1] = unitary3ord(physinds[1], leftbond=Ψbonds[1], rightbond=Ψbonds[2], rightunitary=rightunitary)
    Ψ[end] = unitary3ord(physinds[1], leftbond=Ψbonds[end-1], rightbond=Ψbonds[2], rightunitary=rightunitary)
  end
  return Ψ, Ψbonds, physinds
end

function genΨgauss(;sitenum, physdim, bonddim, withaux=true)
  # Ψbonds = siteinds(bonddim, sitenum + 1)
  Ψbonds = [Index(bonddim, "Ψbond,$i") for i in 1:sitenum+1]
  # physinds = siteinds(physdim, sitenum)
  physinds = [Index(physdim, "physind,$i") for i in 1:sitenum]

  Ψ = [ITensor(reshape([boxmuller() for i in 1:bonddim^2*physdim], (physdim, bonddim, bonddim)),
        physinds[site], Ψbonds[site], Ψbonds[site+1]) for site in 1:sitenum]
  if !withaux
    Ψbonds[1], Ψbonds[end] = Index(1), Index(1)
    Ψ[1] = ITensor(reshape([boxmuller() for i in 1:bonddim*physdim], (physdim, bonddim)), physinds[1], Ψbonds[1], Ψbonds[2])
    Ψ[end] = ITensor(reshape([boxmuller() for i in 1:bonddim*physdim], (physdim, bonddim)), physinds[end], Ψbonds[end-1], Ψbonds[end])
  end
  return Ψ, Ψbonds, physinds
end

function hdens_TIsing(sitenum, physinds, l)
  J = 1; g = 1
  d = dim(physinds[1])

  # hbonds = siteinds(3, sitenum - 1)
  hbonds = [Index(3, "hbond,$i") for i in 1:sitenum-1]
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

function Ψentropies(_Ψ, Ψbonds)
  N = length(_Ψ)
  entropies = Vector{Float64}(undef, N + 1)
  Ψ = deepcopy(_Ψ)

  U, S, V = svd(Ψ[1], Ψbonds[1])
  Ψ[1] = replaceind(S, commonind(U, S) => Ψbonds[1]) * V
  for site in 1:N-1
    U, S, V = svd(Ψ[site] * Ψ[site+1], uniqueinds(Ψ[site], Ψ[site+1]))
    Ψ[site] = U
    Ψ[site+1] = S * V
  end
  U, S, V = svd(Ψ[end], Ψbonds[end])
  Ψ[end] = V * replaceind(S, commonind(S, U) => Ψbonds[end])
  entropies[end] = sings2ent(storage(S))
  for site in N-1:-1:1
    U, S, V = svd(Ψ[site] * Ψ[site+1], uniqueinds(Ψ[site+1], Ψ[site]))
    Ψ[site+1] = δ(Ψbonds[site+1], commonind(S, U)) * U # truncate
    Ψ[site] = V * S * δ(Ψbonds[site+1], commonind(S, U))
    entropies[site+1] = sings2ent(storage(S))
  end
  _, S, _ = svd(Ψ[1], Ψbonds[1])
  entropies[1] = sings2ent(storage(S))

  return entropies
end

function cooldown_seqtrunc(Ψcur, Ψprev, Ψbonds, Dχbonds, physinds, l_h; measents=false, entropies=[])
  N = length(Ψcur)
  #======== left-canonical ========#
  # U, S, V = svd(Ψprev[1] * l_h[1] |> noprime, (Ψbonds[1]))
  # ignore U @ |aux> since nothing operate here and U works same as unit matrix.
  # Ψcur[1] = replaceind(S, commonind(U, S) => Ψbonds[1]) * V
  Ψcur[1] = Ψprev[1] * l_h[1] |> noprime

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

function cooldown_seqtrunc_rev(Ψcur, Ψprev, Ψbonds, Dχbonds, physinds, l_h; measents=false, entropies=[])
  N = length(Ψcur)
  #======== left-canonical ========#
  # U, S, V = svd(Ψprev[1] * l_h[1] |> noprime, (Ψbonds[1]))
  # ignore U @ |aux> since nothing operate here and U works same as unit matrix.
  # Ψcur[1] = replaceind(S, commonind(U, S) => Ψbonds[1]) * V
  Ψcur[end] = Ψprev[end] * l_h[end] |> noprime

  for site in N-1:-1:1
    U, S, V = svd((Ψcur[site] * Ψprev[site+1]) * l_h[site+1] |> noprime, (Dχbonds[site], physinds[site]))
    Dχbonds[site+1] = commonind(U, S)
    Ψcur[site] = U
    Ψcur[site+1] = S * V
  end

  #======== right-canonical ========#
  U, S, V = svd(Ψcur[1], (Ψbonds[1]))
  # ignore U @ |aux> since nothing operate here and contraction is same as unit matrix.
  Ψcur[1] = V * replaceind(S, commonind(S, U) => Ψbonds[1])
  if measents
    entropies[1] = sings2ent(storage(S))
  end

  for site in 1:N-1
    U, S, V = svd(Ψcur[site] * Ψcur[site+1], (Ψbonds[site+2], physinds[site+1]))
    Ψcur[site+1] = δ(Ψbonds[site+1], commonind(S, U)) * U # truncate
    Ψcur[site] = V * S * δ(Ψbonds[site+1], commonind(S, U))
    if measents
      entropies[site+1] = sings2ent(storage(S))
    end
  end
  if measents
    _, S, _ = svd(Ψcur[end], Ψbonds[end])
    entropies[end] = sings2ent(storage(S))
  end
end

function cooldown_unitrunc(Ψcur, Ψprev, Ψbonds, Dχbonds, physinds, l_h; measents=false, entropies=[])
  N = length(Ψcur)
  #======== left-canonical ========#
  Ψcur[1] = Ψprev[1]

  for site in 1:N-1
    U, S, V = svd((Ψcur[site] * Ψprev[site+1]) * l_h[site+1] |> noprime, (Dχbonds[site], physinds[site]))
    Dχbonds[site+1] = commonind(U, S)
    Ψcur[site] = U
    Ψcur[site+1] = S * V
  end

  #======== right-canonical ========#
  U, S, V = svd(Ψcur[end], (Dχbonds[end]))
  # ignore V @ |aux> since nothing operate here and contraction is same as unit matrix.
  Ψcur[end] = V * replaceind(S, commonind(S, U) => Dχbonds[end])
  if measents
    entropies[end] = sings2ent(S * δ(commonind(U, S), Ψbonds[end]) |> storage)
  end

  for site in N-1:-1:1
    U, S, V = svd(Ψcur[site] * Ψcur[site+1], (Dχbonds[site+2], physinds[site+1]))
    Dχbonds[site+1] = commonind(U, S)
    Ψcur[site+1] = U
    Ψcur[site] = V * S
    if measents
      entropies[site+1] = sings2ent(S * δ(Dχbonds[site+1], Ψbonds[site+1]) |> storage)
    end
  end
  if measents
    U, S, V = svd(Ψcur[1], Ψbonds[1])
    entropies[1] = sings2ent(S * δ(commonind(U, S), Ψbonds[1]) |> storage)
  end
  for bond in 2:N # truncate
    Ψcur[bond-1] *= δ(Dχbonds[bond], Ψbonds[bond])
    Ψcur[bond] *= δ(Dχbonds[bond], Ψbonds[bond])
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

function expectedval(Ψ, Ψbonds, physinds, mpo; diag=true, Ψ2=Ψ)
  N = length(Ψ)
  # Ψ and Ψ2 share Ψbonds
  val = prime(Ψ[1], Ψbonds[2], physinds[1]) * mpo[1] * conj(Ψ2[1])
  for site in 2:N-1
    val *= prime(Ψ[site])
    val *= conj(Ψ2[site])
    val *= mpo[site]
  end
  val *= prime(Ψ[end], Ψbonds[end-1], physinds[end])
  val *= conj(Ψ2[end])
  val *= mpo[end]
  if diag
    return real(val[])
  end
  return val[]
end

function plotuβ(;N=16, χ=40, l, kmax=100, counter=0, withaux, cansumplot, seqtrunc=true)
  # N = 16 # number of site
  # χ = 40 # virtual bond dimension of Ψ
  d = 2 # physical dimension
  Ψprev, Ψbonds, physinds = genΨgauss(sitenum=N, bonddim=χ, physdim=d, withaux=withaux)
  norm2₀ = norm2(Ψprev, Ψbonds)
  Ψprev /= norm2₀^inv(2N)
  hdens, l_h = hdens_TIsing(N, physinds, l)

  Ψ = [Ψprev, Vector{ITensor}(undef, N)]
  Dχbonds = Vector{Index}(undef, N + 1)
  Dχbonds[1] = Ψbonds[1]
  Dχbonds[end] = Ψbonds[end]
  u₀ = expectedval(Ψprev, Ψbonds, physinds, hdens)
  uₖs = Vector{Float64}(undef, kmax)
  uₖₖ₊₁s = Vector{ComplexF64}(undef, kmax)
  innerₖₖ₊₁s = Vector{ComplexF64}(undef, kmax)
  ratioₖₖ₊₁s = Vector{Float64}(undef, kmax)

  for k in 1:kmax
    if seqtrunc
      cooldown_seqtrunc(Ψ[k&1+1], Ψ[2-k&1], Ψbonds, Dχbonds, physinds, l_h)
    else
      cooldown_unitrunc(Ψ[k&1+1], Ψ[2-k&1], Ψbonds, Dχbonds, physinds, l_h)
    end
    norm2ₖ = norm2(Ψ[k&1+1], Ψbonds)
    ratioₖₖ₊₁s[k] = sqrt(norm2ₖ)
    println("k=$k, norm2_k = $norm2ₖ")
    Ψ[k&1+1] /= norm2ₖ^inv(2N)
    uₖs[k] = expectedval(Ψ[k&1+1], Ψbonds, physinds, hdens, diag=true, Ψ2=Ψ[k&1+1])
    # for canonical sum
    uₖₖ₊₁s[k] = expectedval(Ψ[k&1+1], Ψbonds, physinds, hdens, diag=false, Ψ2=Ψ[2-k&1])
    innerₖₖ₊₁s[k] = norm2(Ψ[k&1+1], Ψbonds, diag=false, Ψ2=Ψ[2-k&1])
    println("uk = ", uₖs[k])
  end
  println("==== end ====")

  kBT = [0.1:0.1:4;]
  βs = 1 ./ kBT
  uβs = Vector{Float64}(undef, length(kBT))
  for (i, β) in enumerate(βs)
    expval = u₀
    nrm2 = one(ComplexF64)
    coef = one(ComplexF64)
    for k in 0:kmax-1
      coef *= (N * β) / (2k + 1) * ratioₖₖ₊₁s[k+1]
      expval += coef * uₖₖ₊₁s[k+1]
      nrm2 += coef * innerₖₖ₊₁s[k+1]
      coef *= (N * β) / (2k + 2) * ratioₖₖ₊₁s[k+1]
      expval += coef * uₖs[k + 1]
      nrm2 += coef
    end
    uβs[i] = real(expval) / real(nrm2)
  end

  plot!(cansumplot, kBT, uβs, xlabel="kBT", ylabel="E/N", label="l=$l,(No.$counter)", legend = :topleft)

  open("uβ-l=$l,χ=$χ,N=$N,$(seqtrunc ? "seqtrunc" : "unitrunc"),$(withaux ? "withaux" : "noaux"),No$counter.txt", "w") do fp
    content = ""
    for (t, uk) in zip(kBT, uβs)
      content *= "$t $uk\n"
    end
    write(fp, content)
  end

  println("\n\n==== time record ====")
end

function writeΨ(Ψ, filename)
  open(filename, "w") do fp
    content = ""
    for p in Ψ
      for val in storage(p)
        content *= "$val\n"
      end
      content *= "\n"
      write(fp, content)
      content = ""
    end
  end
end

function innerents(_Ψ, _Ψbonds)
  N = length(_Ψ)
  entropies = Vector{Float64}(undef, N ÷ 2)
  Ψ = deepcopy(_Ψ)
  Ψbonds = deepcopy(_Ψbonds)
  pχbonds = deepcopy(Ψbonds)

  for count in 1:N÷2-1
    U, S1, V = svd(Ψ[count] * Ψ[count+1], uniqueinds(Ψ[count], Ψ[count+1]))
    Ψ[count] = U
    Ψ[count+1] = S1 * V
    pχbonds[count+1] = commonind(U, S1)
    U, S2, V = svd(Ψ[N-count] * Ψ[N+1-count], uniqueinds(Ψ[N+1-count], Ψ[N-count]))
    Ψ[N+1-count] = U
    Ψ[N-count] = S2 * V
    pχbonds[N+1-count] = commonind(U, S2)
  end

  println("toward-center unitarized.")
  for (i, p) in enumerate(Ψ)
    println("\n\n$i : ")
    display(p)
  end

  bulk = 1
  for count in 0:N÷2-1
    if count == 0
      Ψ[N÷2] *= δ(pχbonds[N÷2], Ψbonds[N÷2])
      Ψ[N÷2+1] *= δ(pχbonds[N÷2+2], Ψbonds[N÷2+2])
    elseif count == N ÷ 2 - 1
      Ψ[1] *= δ(pχbonds[2], Ψbonds[2])
      Ψ[end] *= δ(pχbonds[end-1], Ψbonds[end-1])
    else
      Ψ[N÷2-count] *= δ(pχbonds[N÷2-count+1], Ψbonds[N÷2-count+1])
      Ψ[N÷2-count] *= δ(pχbonds[N÷2-count], Ψbonds[N÷2-count])
      Ψ[N÷2+1+count] *= δ(pχbonds[N÷2+1+count], Ψbonds[N÷2+1+count])
      Ψ[N÷2+1+count] *= δ(pχbonds[N÷2+2+count], Ψbonds[N÷2+2+count])
    end
    bulk *= Ψ[N÷2-count]
    bulk *= prime(Ψ[N÷2-count], Ψbonds[N÷2-count], Ψbonds[N÷2-count+1]) |> conj
    bulk *= Ψ[N÷2+1+count]
    bulk *= prime(Ψ[N÷2+1+count], Ψbonds[N÷2+2+count], Ψbonds[N÷2+1+count]) |> conj
    mbulk = bulk * combiner(Ψbonds[N÷2-count], Ψbonds[N÷2+2+count]) * combiner(Ψbonds[N÷2-count]', Ψbonds[N÷2+2+count]')
    println("\nmbulk dim: " )
    display(mbulk)
    λ = eigvals(mbulk |> matrix) |> real
    entropies[count+1] = -sum(λ ./ sum(λ) .|> x -> abs(x) * log(abs(x)))
  end

  return entropies
end

function plotents(;l, N, χ, withaux, seqtrunc=true, meastemps=[200:200:1600;], measinnerents=false)
  d = 2 # physical dimension
  kmax = maximum(meastemps)
  Ψprev, Ψbonds, physinds = genΨgauss(sitenum=N, bonddim=χ, physdim=d, withaux=withaux)
  Ψprev /= norm2(Ψprev, Ψbonds)^inv(2N)
  _, l_h = hdens_TIsing(N, physinds, l)

  Ψcouple = [Ψprev, Vector{ITensor}(undef, N)]
  Dχbonds = Vector{Index}(undef, N + 1)
  Dχbonds[1] = Ψbonds[1]
  Dχbonds[end] = Ψbonds[end]
  entsfortemp = Dict()
  innerentsfortemp = Dict()
  entropies = Vector{Float64}(undef, N + 1)

  entsfortemp["0"] = Ψentropies(Ψprev, Ψbonds)
  if measinnerents
    innerentsfortemp["0"] = innerents(Ψcouple[1], Ψbonds)
  end
  cooldown = seqtrunc ? cooldown_seqtrunc : cooldown_unitrunc

  for k in 1:kmax
    println("----- k = $k -----")
    if k in meastemps
      cooldown(Ψcouple[k&1+1], Ψcouple[2-k&1], Ψbonds, Dχbonds, physinds, l_h, measents=true, entropies=entropies)
      entsfortemp["$k"] = deepcopy(entropies)
      if measinnerents
        innerentsfortemp["$k"] = innerents(Ψcouple[k&1+1], Ψbonds)
      end
      writeΨ(Ψcouple[k&1+1], "Ψ-l=$l,k=$k,N=$N,χ=$χ,$(seqtrunc ? "seqtrunc" : "unitrunc"),$(withaux ? "withaux" : "noaux").txt")
    else
      cooldown(Ψcouple[k&1+1], Ψcouple[2-k&1], Ψbonds, Dχbonds, physinds, l_h)
    end
    # Ψnorm2 = norm2(Ψcouple[k&1+1], Ψbonds)
    # println("k=$k, Ψnorm2 = $Ψnorm2")
    # println("k=$k")
    # Ψcouple[k&1+1] /= Ψnorm2^inv(2N)
  end
  println("==== end ====")

  open("entropies-l=$l,N=$N,χ=$χ,$(seqtrunc ? "seqtrunc" : "unitrunc"),$(withaux ? "withaux" : "noaux").txt", "w") do fp
    content = ""
    for item in entsfortemp
      content *= item.first * "\n"
      for ee in item.second
        content *= "$ee\n"
      end
      content *= "\n"
    end
    write(fp, content)
  end
  plot()
  for item in entsfortemp
    plot!([0:N;], item.second, xlabel = "i", ylabel = "S_i", label = "k=$(item.first)")
  end
  savefig("entropies-l=$l,N=$N,χ=$χ,$(seqtrunc ? "seqtrunc" : "unitrunc"),$(withaux ? "withaux" : "noaux").png")

  if measinnerents
    open("innerents-l=$l,N=$N,χ=$χ,$(seqtrunc ? "seqtrunc" : "unitrunc"),$(withaux ? "withaux" : "noaux").txt", "w") do fp
      content = ""
      for item in innerentsfortemp
        content *= item.first * "\n"
        for ee in item.second
          content *= "$ee\n"
        end
        content *= "\n"
      end
      write(fp, content)
    end
    plot()
    for item in innerentsfortemp
      plot!([2:2:N;], item.second, xlabel = "i", ylabel = "S_i", label = "k=$(item.first)")
    end
    savefig("innerents-l=$l,N=$N,χ=$χ,$(seqtrunc ? "seqtrunc" : "unitrunc"),$(withaux ? "withaux" : "noaux").png")
  end

  println("\n\n==== time record ====")
end

function plotuβs_samel()
  cansumplot = plot()
  withaux = true
  seqtrunc = true
  l = 5
  kmax = 400
  χ = 20
  N = 16
  for i in 1:3
    @time plotuβ(N=N, χ=χ, l=l, kmax=200, counter=i, withaux=withaux, cansumplot=cansumplot, seqtrunc=seqtrunc)
  end
  plot(cansumplot)
  savefig("uβ-l=$l,kmax=$kmax,χ=$χ,N=$N,withaux=$withaux,seqtrunc=$seqtrunc.png")
end

function plotuβs_forl()
  cansumplot = plot()
  kmax = 200
  seqtrunc = true
  withaux = true
  for l in [5]
    plotuβ(l=l, kmax=kmax, withaux=withaux, cansumplot=cansumplot, seqtrunc=seqtrunc)
  end
  plot(cansumplot)
  savefig("uβ-forl,kmax=$kmax,χ=40,N=16,withaux=$withaux,seqtrunc=$seqtrunc.png")
end

@time plotents(l=5, N=64, χ=40, withaux=true, meastemps=[0,3], measinnerents=true)
# @time plotuβs_samel()
