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
  χ = max(dim(leftbond), dim(rightbond))
  d = dim(physind)
  q, _ = reshape([boxmuller() for i in 1:(χ*d)^2], (χ * d, χ * d)) |> qr
  u = reshape(q, (d, χ, d, χ))
  if !rightunitary
    return ITensor(u[:, 1:dim(leftbond), 1, 1:dim(rightbond)], physind, leftbond, rightbond) # leftunitary
  end
  return ITensor(u[1, 1:dim(leftbond), :, 1:dim(rightbond)], leftbond, physind, rightbond) # rightunitary
end

function genΨ(;sitenum, physdim, bonddim, withaux=true, rightunitary=false)
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

function genΨnocanonical(;sitenum, physdim, bonddim, withaux=true)
  Ψbonds = siteinds(bonddim, sitenum + 1)
  physinds = siteinds(physdim, sitenum)

  Ψ = [ITensor(reshape([boxmuller() for i in 1:bonddim^2*physdim], (physdim, bonddim, bonddim)),
        physinds[site], Ψbonds[site], Ψbonds[site+1]) for site in 1:sitenum]
  if !withaux
    Ψbonds[1], Ψbonds[end] = Index(1), Index(1)
    Ψ[1] = ITensor(reshape([boxmuller() for i in 1:bonddim*physdim], (physdim, bonddim)), physinds[1], Ψbonds[1], Ψbonds[2])
    Ψ[end] = ITensor(reshape([boxmuller() for i in 1:bonddim*physdim], (physdim, bonddim)), physinds[end], Ψbonds[end-1], Ψbonds[end])
  end
  return Ψ, Ψbonds, physinds
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

# λ : array of singular values(not necessarily normalized)
function entropy(λ)
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
  entropies[end] = entropy(storage(S))
  for site in N-1:-1:1
    U, S, V = svd(Ψ[site] * Ψ[site+1], uniqueinds(Ψ[site+1], Ψ[site]))
    Ψ[site+1] = δ(Ψbonds[site+1], commonind(S, U)) * U # truncate
    Ψ[site] = V * S * δ(Ψbonds[site+1], commonind(S, U))
    entropies[site+1] = entropy(storage(S))
  end
  _, S, _ = svd(Ψ[1], Ψbonds[1])
  entropies[1] = entropy(storage(S))

  return entropies
end

function cooldown_seqtrunc(Ψk, Ψprev, Ψbonds, Dχbonds, physinds, l_h; measentropies=false, entropies=[])
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
  if measentropies
    entropies[end] = entropy(storage(S))
  end

  for site in N-1:-1:1
    U, S, V = svd(Ψk[site] * Ψk[site+1], (Ψbonds[site+2], physinds[site+1]))
    Ψk[site+1] = δ(Ψbonds[site+1], commonind(S, U)) * U # truncate
    Ψk[site] = V * S * δ(Ψbonds[site+1], commonind(S, U))
    if measentropies
      entropies[site+1] = entropy(storage(S))
    end
  end
  if measentropies
    _, S, _ = svd(Ψk[1], Ψbonds[1])
    entropies[1] = entropy(storage(S))
  end
end

function cooldown_unitrunc(Ψk, Ψprev, Ψbonds, Dχbonds, physinds, l_h; measentropies=false, entropies=[])
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
  U, S, V = svd(Ψk[end], (Dχbonds[end]))
  # ignore V @ |aux> since nothing operate here and contraction is same as unit matrix.
  Ψk[end] = V * replaceind(S, commonind(S, U) => Dχbonds[end])
  if measentropies
    entropies[end] = entropy(storage(S * δ(commonind(U, S), Ψbonds[end])))
  end

  for site in N-1:-1:1
    U, S, V = svd(Ψk[site] * Ψk[site+1], (Dχbonds[site+2], physinds[site+1]))
    Dχbonds[site+1] = commonind(U, S)
    Ψk[site+1] = U
    Ψk[site] = V * S
    if measentropies
      entropies[site+1] = entropy(storage(S * δ(Dχbonds[site+1], Ψbonds[site+1])))
    end
  end
  if measentropies
    U, S, V = svd(Ψk[1], Ψbonds[1])
    entropies[1] = entropy(storage(S * δ(commonind(U, S), Ψbonds[1])))
  end
  for bond in 2:N # truncate
    Ψk[bond-1] *= δ(Dχbonds[bond], Ψbonds[bond])
    Ψk[bond] *= δ(Dχbonds[bond], Ψbonds[bond])
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

function plotenergydensity(;l, kmax=100, counter=0, pall=plot(), psub=plot(), savefigure=true, withaux=true)
  N = 16 # number of site
  χ = 40 # virtual bond dimension of Ψ
  d = 2 # physical dimension
  #kmax: num to cool down
  Ψprev, Ψbonds, physinds = genΨnocanonical(sitenum=N, bonddim=χ, physdim=d, withaux=withaux)
  hamdens, l_h = hamdens_transverseising(N, physinds, l)

  Ψ = [Ψprev, Vector{ITensor}(undef, N)]
  Dχbonds = Vector{Index}(undef, N + 1)
  Dχbonds[1] = Ψbonds[1]
  Dχbonds[end] = Ψbonds[end]
  uₖs = Vector{Float64}(undef, kmax)

  for k in 1:kmax
    cooldown_seqtrunc(Ψ[k&1+1], Ψ[2-k&1], Ψbonds, Dχbonds, physinds, l_h)
    Ψnorm2 = norm2(Ψ[k&1+1], Ψbonds)
    println("k=$k, Ψnorm2 = $Ψnorm2")
    uₖs[k] = expectedvalue(Ψ[k&1+1], Ψbonds, physinds, hamdens, Ψnorm2)
    println("uk = ", uₖs[k])
    Ψ[k&1+1] /= Ψnorm2^inv(2N)
  end

  println("==== end ====")
  kBT = [N * (l - uₖs[k]) / 2k for k in 1:kmax]
  open("energy-density-l=$l,χ=$χ,N=$N,noncanon,seqtrunc,No$counter,$(withaux ? "" : "noaux").txt", "w") do fp
    content = ""
    for (t, uk) in zip(kBT, uₖs)
      content *= "$t $uk\n"
    end
    write(fp, content)
  end
  scatter!(pall, kBT, uₖs, xlabel="k_BT", ylabel="u_k", label="l=$l,(No.$counter)")
  if savefigure
    savefig("energy-density-l=$l,χ=$χ,N=$N,noncanon,seqtrunc,No$counter,$(withaux ? "" : "noaux").png")
  end
  scatter!(psub, kBT, uₖs, xlabel="k_BT", ylabel="u_k", label="l=$l,(No.$counter)", xlims=(0,4))
  if savefigure
    savefig("energy-density-l=$l,χ=$χ,N=$N,noncanon,seqtrunc,No$counter,$(withaux ? "" : "noaux"),sub.png")
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

function plotentropies(;l)
  N = 64 # number of site
  χ = 40 # virtual bond dimension of Ψ
  d = 2 # physical dimension
  kmax = 200 # num to cool down
  Ψprev, Ψbonds, physinds = genΨnocanonical(sitenum=N, bonddim=χ, physdim=d)
  _, l_h = hamdens_transverseising(N, physinds, l)

  Ψ = [Ψprev, Vector{ITensor}(undef, N)]
  Dχbonds = Vector{Index}(undef, N + 1)
  Dχbonds[1] = Ψbonds[1]
  Dχbonds[end] = Ψbonds[end]
  multitempentropies = Dict()
  entropies = Vector{Float64}(undef, N + 1)
  # meastemperatures = [200:200:1600;]
  meastemperatures = [0:40:200;]

  multitempentropies["0"] = Ψentropies(Ψprev, Ψbonds)

  for k in 1:kmax
    if k in meastemperatures
      cooldown_unitrunc(Ψ[k&1+1], Ψ[2-k&1], Ψbonds, Dχbonds, physinds, l_h, measentropies=true, entropies=entropies)
      multitempentropies["$k"] = deepcopy(entropies)
      writeΨ(Ψ[k&1+1], "Ψ-l=$l,k=$k,N=$N,χ=$χ,noncanon,unitruncate.txt")
    else
      cooldown_unitrunc(Ψ[k&1+1], Ψ[2-k&1], Ψbonds, Dχbonds, physinds, l_h)
    end
    Ψnorm2 = norm2(Ψ[k&1+1], Ψbonds)
    println("k=$k, Ψnorm2 = $Ψnorm2")
    Ψ[k&1+1] /= Ψnorm2^inv(2N)
  end

  println("==== end ====")
  open("entropies-l=$l.txt", "w") do fp
    content = ""
    for item in multitempentropies
      content *= item.first * "\n"
      for ee in item.second
        content *= "$ee\n"
      end
      content *= "\n"
    end
    write(fp, content)
  end
  for item in multitempentropies
    plot!([0:N;], item.second, xlabel = "i", ylabel = "S_i", label = "k=$(item.first)")
  end
  savefig("entanglemententropy-l=$l,N=$N,χ=$χ,noncanon,unitrunc.png")
  println("\n\n==== time record ====")
end

function plotmultienergydensities_samel()
  pall = plot()
  psub = plot()
  withaux = false
  for i in 1:5
    @time plotenergydensity(l=8, kmax=100, counter=i, pall=pall, psub=psub, savefigure=false, withaux=withaux)
  end
  plot(pall)
  savefig("energy-density-l=8,χ=40,N=16,noncanon,seqtrunc,$(withaux ? "" : "noaux"),all.png")
  plot(psub)
  savefig("energy-density-l=8,χ=40,N=16,noncanon,seqtrunc,$(withaux ? "" : "noaux"),sub.png")
end

function plotmultienergydensities_forl()
  pall = plot()
  psub = plot()
  for l in [2, 4, 8, 16]
    plotenergydensity(l=l, kmax=100, counter=0, pall=pall, psub=psub, savefigure=false)
  end
  plot(pall)
  savefig("energy-density-forl,χ=40,N=16,noncanon,seqtrunc,all.png")
  plot(psub)
  savefig("energy-density-forl,χ=40,N=16,noncanon,seqtrunc,sub.png")
end

@time plotmultienergydensities_samel()