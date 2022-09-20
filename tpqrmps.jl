using ITensors
using JLD2
using LinearAlgebra
using Plots
using Printf
using Random
using Base.Threads
import Plots.plot
import Plots.plot!
@show nthreads()
pyplot()

"make Ψ with gauss distribution."
function genΨgauss(;sitenum, physdim, bonddim, edgedim=bonddim, seed=-1)
  Ψbonds = siteinds(bonddim, sitenum + 1)
  physinds = siteinds(physdim, sitenum)
  Ψbonds[1], Ψbonds[end] = Index(edgedim), Index(edgedim)
  Ψ = [ITensor(randn(MersenneTwister(), ComplexF64, (physdim, dim(Ψbonds[site]), dim(Ψbonds[site+1]))),
        physinds[site], Ψbonds[site], Ψbonds[site+1]) for site in 1:sitenum]
  return Ψ, Ψbonds, physinds
end

"make hamiltonian density and l - h of transverse ising model."
function hdens_TIsing(sitenum, physinds, l; J=1, g=1)
  d = dim(physinds[1])

  hbonds = [Index(3, "hbond,$i") for i in 1:sitenum-1]
  leftmold = zeros(d, 3, d)
  rightmold = zeros(d, 3, d)
  leftmold[:, 1, :] = rightmold[:, 3, :] = I(2)
  leftmold[:, 2, :] = rightmold[:, 2, :] = sqrt(J / sitenum) * [1 0;0 -1]
  leftmold[:, 3, :] = rightmold[:, 1, :] = -(g / sitenum) * [0 1;1 0]
  h_left = ITensor(leftmold, physinds[1], hbonds[1], physinds[1]')
  h_right = ITensor(rightmold, physinds[end], hbonds[end], physinds[end]')

  middlemold = zeros(3, d, 3, d)
  middlemold[1, :, 1, :] = I(2)
  middlemold[3, :, 3, :] = I(2)
  middlemold[1, :, 2, :] = middlemold[2, :, 3, :] = sqrt(J / sitenum) * [1 0;0 -1]
  middlemold[1, :, 3, :] = -(g / sitenum) * [0 1;1 0]

  hdens = [ITensor(middlemold, hbonds[i-1], physinds[i], hbonds[i], physinds[i]') for i in 2:sitenum-1]
  pushfirst!(hdens, h_left)
  push!(hdens, h_right)

  rightmold2 = deepcopy(rightmold)
  rightmold2[:, 1, :] -= l * I(2) # I ⊗ ⋯ ⊗ (-l*I+σx)
  l_h_right = ITensor(-rightmold2, physinds[end], hbonds[end], physinds[end]') # coef "-1" smashed in here

  l_h = deepcopy(hdens)
  l_h[end] = l_h_right
  return (hdens, l_h)
end

"identity mpo"
function mpoid(sitenum, physinds, l)
  return [ITensor(I(dim(physinds[i])), physinds[i], physinds[i]') for i in 1:sitenum]
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

  for site in 1:N-1
    Q, R = qr(Ψ[site] * Ψ[site+1], uniqueinds(Ψ[site], Ψ[site+1]))
    Ψ[site] = Q
    Ψ[site+1] = R
  end
  _, S, _ = svd(Ψ[end], Ψbonds[end])
  # Ψ[end] = V * replaceind(S, commonind(S, U) => Ψbonds[end])
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

function cooldown_seqtrunc(Ψcur, Ψprev, Ψbonds, Dχbonds, physinds, l_h; entropies=[], rev=false)
  N = length(Ψcur)
  measents = entropies != []
  at(arr, ind) = rev ? getindex(arr, reverseind(arr, ind)) : getindex(arr, ind)
  set!(arr, val, ind) = rev ? setindex!(arr, val, reverseind(arr, ind)) : setindex!(arr, val, ind)

  #======== left-canonical ========#
  set!(Ψcur, at(Ψprev, 1) * at(l_h, 1) |> noprime, 1)
  for site in 1:N-1
    Q, R = qr((at(Ψcur, site) * at(Ψprev, site + 1)) * at(l_h, site + 1) |> noprime, (at(Dχbonds, site), at(physinds, site)))
    set!(Dχbonds, commonind(Q, R), site + 1)
    set!(Ψcur, Q, site)
    set!(Ψcur, R, site + 1)
  end

  #======== right-canonical ========#
  if measents
    _, S, _ = svd(at(Ψcur, N), at(Ψbonds, N + 1))
    set!(entropies, sings2ent(storage(S)), N + 1)
  end

  for site in N-1:-1:1
    U, S, V = svd(at(Ψcur, site) * at(Ψcur, site + 1), (at(Ψbonds, site + 2), at(physinds, site + 1)))
    set!(Ψcur, δ(at(Ψbonds, site + 1), commonind(S, U)) * U, site + 1) # truncate
    set!(Ψcur, V * S * δ(at(Ψbonds, site + 1), commonind(S, U)), site)

    if measents
      set!(entropies, sings2ent(storage(S)), site + 1)
    end
  end
  if measents
    _, S, _ = svd(at(Ψcur, 1), at(Ψbonds, 1))
    set!(entropies, sings2ent(storage(S)), 1)
  end
end

function norm2(Ψ, Ψbonds; Ψ2=Ψ)
  diag = Ψ === Ψ2
  N = length(Ψ)
  leftcont = Threads.@spawn let at(arr, ind) = getindex(arr, ind)
    left = prime(at(Ψ, 1), at(Ψbonds, 2)) * conj(at(Ψ2, 1))
    for site in 2:N÷2
      left *= prime(at(Ψ, site), at(Ψbonds, site), at(Ψbonds, site + 1))
      left *= conj(at(Ψ2, site))
    end
    left
  end
  rightcont = Threads.@spawn let at(arr, ind) = getindex(arr, reverseind(arr, ind))
    right = prime(at(Ψ, 1), at(Ψbonds, 2)) * conj(at(Ψ2, 1))
    for site in 2:N÷2
      right *= prime(at(Ψ, site), at(Ψbonds, site), at(Ψbonds, site + 1))
      right *= conj(at(Ψ2, site))
    end
    right
  end
  ret = fetch(leftcont) * fetch(rightcont)
  return diag ? real(ret[]) : ret[]
end

function expectedval(Ψ, Ψbonds, physinds, mpo; Ψ2=Ψ)
  diag = Ψ === Ψ2
  N = length(Ψ)
  # Ψ and Ψ2 share Ψbonds
  leftcont = Threads.@spawn let at(arr, ind) = getindex(arr, ind)
    left = prime(at(Ψ, 1), at(Ψbonds, 2), at(physinds, 1)) * at(mpo, 1) * conj(at(Ψ2, 1))
    for site in 2:N÷2
      left *= prime(at(Ψ, site))
      left *= conj(at(Ψ2, site))
      left *= at(mpo, site)
    end
    left
  end
  rightcont = Threads.@spawn let at(arr, ind) = getindex(arr, reverseind(arr, ind))
    right = prime(at(Ψ, 1), at(Ψbonds, 2), at(physinds, 1)) * at(mpo, 1) * conj(at(Ψ2, 1))
    for site in 2:N÷2
      right *= prime(at(Ψ, site))
      right *= conj(at(Ψ2, site))
      right *= at(mpo, site)
    end
    right
  end
  ret = fetch(leftcont) * fetch(rightcont)
  return diag ? real(ret[]) : ret[]
end

infos(;l, χ, χaux, N, kmax, seqtrunc, rev, repnum=-1) = "l=$l,N=$N,χ=$χ,χaux=$χaux,kmax=$kmax,trunc=$(seqtrunc ? "seq" : "uni"),rev=$rev$(repnum>0 ? ",repnum=$repnum" : "")"

function plotuβ(;N=16, χ=40, l, kmax=100, counter=0, cansumplot, seqtrunc=true, edgedim=χ, rev=false, savedata=false)
  infost = infos(l=l, N=N, χ=χ, χaux=edgedim, kmax=kmax, seqtrunc=seqtrunc, rev=rev)
  d = 2 # physical dimension
  Ψprev, Ψbonds, physinds = genΨgauss(sitenum=N, bonddim=χ, physdim=d, edgedim=edgedim)
  norm2₀ = norm2(Ψprev, Ψbonds)
  Ψprev /= norm2₀^inv(2N)
  hdens, l_h = hdens_TIsing(N, physinds, l, J=1, g=1)

  Ψ = [Ψprev, Vector{ITensor}(undef, N)]
  Dχbonds = Vector{Index}(undef, N + 1)
  Dχbonds[1], Dχbonds[end] = Ψbonds[1], Ψbonds[end]
  u₀ = expectedval(Ψprev, Ψbonds, physinds, hdens)
  uₖs = Vector{Float64}(undef, kmax)
  uₖₖ₊₁s = Vector{ComplexF64}(undef, kmax)
  innerₖₖ₊₁s = Vector{ComplexF64}(undef, kmax)
  ratioₖₖ₊₁s = Vector{Float64}(undef, kmax)

  println()
  for k in 1:kmax
    if seqtrunc
      cooldown_seqtrunc(Ψ[k&1+1], Ψ[2-k&1], Ψbonds, Dχbonds, physinds, l_h, rev=rev)
    # else
    #   cooldown_unitrunc(Ψ[k&1+1], Ψ[2-k&1], Ψbonds, Dχbonds, physinds, l_h, rev=rev)
    end
    norm2ₖ = norm2(Ψ[k&1+1], Ψbonds)
    ratioₖₖ₊₁s[k] = sqrt(norm2ₖ |> real)
    println("k=$k, norm2_k = $norm2ₖ")
    Ψ[k&1+1] /= real(norm2ₖ)^inv(2N)
    task1 = Threads.@spawn expectedval(Ψ[k&1+1], Ψbonds, physinds, hdens)
    task2 = Threads.@spawn expectedval(Ψ[k&1+1], Ψbonds, physinds, hdens, Ψ2=Ψ[2-k&1])
    task3 = Threads.@spawn norm2(Ψ[k&1+1], Ψbonds, Ψ2=Ψ[2-k&1])
    uₖs[k] = fetch(task1)
    uₖₖ₊₁s[k] = fetch(task2)
    innerₖₖ₊₁s[k] = fetch(task3)
    println("\tuk = ", uₖs[k])
    print("\t<k|k+1> = $(innerₖₖ₊₁s[k])\r\033[2A")
  end
  println("\n\n\n==== end ====")

  kBT = [0.1:0.1:4;]
  βs = 1 ./ kBT
  uβs = Vector{Float64}(undef, length(kBT))
  for (i, β) in enumerate(βs)
    expval = u₀
    nrm2 = one(BigFloat)
    coef = one(BigFloat)
    for k in 0:kmax-1
      coef *= (N * β) / (2k + 1) * ratioₖₖ₊₁s[k+1]
      expval += coef * uₖₖ₊₁s[k+1]
      nrm2 += coef * innerₖₖ₊₁s[k+1]

      coef *= (N * β) / (2k + 2) * ratioₖₖ₊₁s[k+1]
      expval += coef * uₖs[k + 1]
      nrm2 += coef
    end
    if nrm2 == 0
      println("nrm2 == 0")
    end
    uβs[i] = real(expval) / real(nrm2)
  end

  plot!(cansumplot, kBT, uβs, xlabel="kBT", ylabel="E/N", label="l=$l,(No.$counter)", legend=:outerleft, title=infost)
  # plot!(cansumplot, kBT, uβs_ex, xlabel="kBT", ylabel="E/N", label="l=$l,(No.$counter)-ex", legend = :outerleft)

  if savedata
    open("uβ-$infost,No$counter.txt", "w") do fp
      # content = ""
      # for (t, uk) in zip(kBT, uβs)
      #   content *= "$t $uk\n"
      # end
      # write(fp, content)
      Base.print_array(fp, hcat(kBT, uβs))
    end

    open("ukBT-$infost,No$counter.txt", "w") do fp
      # content = ""
      # for (t, uk) in zip([N * (l - uₖs[k]) / 2k for k in 1:kmax], uₖs)
      #   content *= "$t $uk\n"
      # end
      # write(fp, content)
      Base.print_array(fp, hcat([N * (l - uₖs[k]) / 2k for k in 1:kmax], uₖs))
    end
  end

  println("\n\n==== time record ====")
  return kBT, uβs, [N * (l - uₖs[k]) / 2k for k in 1:kmax], uₖs
end

function writeΨ(Ψ, filename)
  # open(filename, "w") do fp
  #   content = ""
  #   for p in Ψ
  #     for val in storage(p)
  #       content *= "$val\n"
  #     end
  #     content *= "\n"
  #     write(fp, content)
  #     content = ""
  #   end
  # end
  jldopen(filename * ".jld2", "w") do file
    write(file, "Ψ", Ψ)
  end
end

function innerents(_Ψ, _Ψbonds)
  N = length(_Ψ)

  Ψ = deepcopy(_Ψ)
  Ψbonds = deepcopy(_Ψbonds)
  pχbonds = deepcopy(Ψbonds)

  for count in 1:N÷2-1
    Q, R = qr(Ψ[count] * Ψ[count+1], uniqueinds(Ψ[count], Ψ[count+1]))
    Ψ[count] = Q
    Ψ[count+1] = R
    pχbonds[count+1] = commonind(Q, R)
    Q, R = qr(Ψ[N-count] * Ψ[N+1-count], uniqueinds(Ψ[N+1-count], Ψ[N-count]))
    Ψ[N+1-count] = Q
    Ψ[N-count] = R
    pχbonds[N+1-count] = commonind(Q, R)
  end

  bulk = 1
  enttasks = Vector{Task}(undef, N ÷ 2)
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
    let mbulk = mbulk
      enttasks[count+1] = Threads.@spawn begin
        λ = eigvals(mbulk |> matrix) |> real
        nrm = sum(λ)
        if nrm == 0
          nrm = 1
        end
        -sum(λ ./ sum(λ) .|> x -> abs(x) * log(abs(x)))
      end
    end
  end

  return fetch.(enttasks)
end

function measents(;l, N, χ, seqtrunc=true, Ts=[200:200:1600;], inents=false, rev=false, edgedim=χ, saveΨs=false)
  d = 2 # physical dimension
  kmax = maximum(Ts)
  infost = infos(l=l, N=N, χ=χ, χaux=edgedim, kmax=kmax, seqtrunc=seqtrunc, rev=rev)
  Ψprev, Ψbonds, physinds = genΨgauss(sitenum=N, bonddim=χ, physdim=d, edgedim=edgedim)
  Ψprev /= norm2(Ψprev, Ψbonds)^inv(2N)
  _, l_h = hdens_TIsing(N, physinds, l)

  Ψcouple = [Ψprev, Vector{ITensor}(undef, N)]
  Dχbonds = Vector{Index}(undef, N + 1)
  Dχbonds[1], Dχbonds[end] = Ψbonds[1], Ψbonds[end]
  entsfortemp = [0:N;]
  innerentsfortemp = [2:2:N;]
  entropies = Vector{Float64}(undef, N + 1)

  entsfortemp = hcat(entsfortemp, Ψentropies(Ψprev, Ψbonds))
  if inents
    innerentsfortemp = hcat(innerentsfortemp, innerents(Ψcouple[1], Ψbonds))
  end
  cooldown = seqtrunc ? cooldown_seqtrunc : cooldown_seqtrunc

  for k in 1:kmax
    if k in Ts
      cooldown(Ψcouple[k&1+1], Ψcouple[2-k&1], Ψbonds, Dχbonds, physinds, l_h, entropies=entropies, rev=rev)
      entsfortemp = hcat(entsfortemp, entropies)
      if inents
        innerentsfortemp = hcat(innerentsfortemp, innerents(Ψcouple[k&1+1], Ψbonds))
      end
      if saveΨs
        jldopen("Ψ-$infost.jld2", "a+") do file
          write(file, "$k", Ψcouple[k&1+1])
        end
      end
    else
      cooldown(Ψcouple[k&1+1], Ψcouple[2-k&1], Ψbonds, Dχbonds, physinds, l_h, rev=rev)
    end
    Ψnorm2 = norm2(Ψcouple[k&1+1], Ψbonds)
    print("k=$k, Ψnorm2 = $Ψnorm2\r")
    Ψcouple[k&1+1] /= Ψnorm2^inv(2N)
  end
  println("==== end ====")

  open("entropies-$infost.dat", "w") do fp
    print(fp, "T")
    Base.print_array(fp, Ts')
    println(fp, "\n")
    Base.print_array(fp, entsfortemp)
  end
  plot()
  for i in eachindex(Ts)
    plot!(entsfortemp[:, 1], entsfortemp[:, 1+i], xlabel = "i", ylabel = "S_i", label = "k=$(Ts[i])", legend = :outerright, title=infost, size=(1200, 400))
  end
  savefig("entropies-$infost.png")

  if inents
    open("innerents-$infost.dat", "w") do fp
      print(fp, "T")
      Base.print_array(fp, Ts')
      println(fp, "\n")
      Base.print_array(fp, innerentsfortemp)
    end
    plot()
    for i in eachindex(Ts)
      plot!(innerentsfortemp[:, 1], innerentsfortemp[:, 1+i], xlabel = "i", ylabel = "S_i", label = "k=$(Ts[i])", legend = :outerright, title=infost, size=(1200, 400))
    end
    savefig("innerents-$infost.png")
  end

  println("\n\n==== time record ====")
end

function repcalcuβs(;l, kmax, χ, N, repnum, seqtrunc, edgedim=χ, rev=false, saveplot=true, savedata=true)
  cansumplot = plot()
  uβss = Vector{Vector{Float64}}(undef, repnum)
  kBT = []
  infost = infos(l=l, N=N, χ=χ, χaux=edgedim, kmax=kmax, seqtrunc=seqtrunc, rev=rev, repnum=repnum)
  for i in 1:repnum
    @time kBT, uβss[i], _, _ = plotuβ(N=N, χ=χ, l=l, kmax=kmax, counter=i, cansumplot=cansumplot, seqtrunc=seqtrunc, edgedim=edgedim, rev=rev)
  end
  if saveplot
    plot(cansumplot)
    savefig("uβ-$infost.png")
  end
  aves = zeros(Float64, length(uβss[1]))
  var = zeros(Float64, length(uβss[1]))
  for i in 1:repnum
    for j in 1:length(uβss[1])
      aves[j] += uβss[i][j]
    end
  end
  aves /= repnum
  for i in 1:repnum
    for j in 1:length(uβss[1])
      var[j] += (aves[j] - uβss[i][j]) ^ 2
    end
  end
  var /= repnum - 1 # unbiased variance
  ses = sqrt.(var) ./ sqrt(repnum) # standard error
  # plot(kBT, aves, xlabel="kBT", ylabel="E/N average", yerror=ses, legend=false)
  # savefig("ave_uβ-l=$l,kmax=$kmax,χ=$χ,N=$N,seqtrunc=$seqtrunc,repnum=$repnum,edgedim=$edgedim.png")

  open("uβave-$infost.txt", "w") do fp
    # content = ""
    # for (t, ave, se) in zip(kBT, aves, ses)
    #   content *= "$t $ave $se\n"
    # end
    # write(fp, content)
    Base.print_array(fp, hcat(kBT, aves, ses))
  end
  return kBT, aves, ses
end

# function plotuβs_forl()
#   cansumplot = plot()
#   kmax = 200
#   seqtrunc = true
#   withaux = true
#   N = 16
#   χ = 20
#   for l in [2,8,32]
#     plotuβ(N=N, χ=χ, l=l, kmax=kmax, cansumplot=cansumplot, seqtrunc=seqtrunc)
#   end
#   plot(cansumplot)
#   savefig("uβ-forl,kmax=$kmax,χ=$χ,N=$N,withaux=$withaux,seqtrunc=$seqtrunc.png")
# end

# function loadstartmeasents(filename, lastk)
#   Ψ = load(filename, lastk)

# end

# @time measents(l=5, N=64, χ=40, seqtrunc=true, Ts=[0:200:1200;], inents=true, rev=false, saveΨs=true, edgedim=1)
# @time plotuβs_samel(l=5, kmax=500, χ=40, N=16, repnum=5, seqtrunc=true, edgedim=40)

function cmpse_forχaux()
  aveplot = plot()
  sesplot = plot()
  χauxs = [1, 5, 10, 20]
  l = 5; kmax = 500; χ = 5; N = 8; repnum = 50; seqtrunc = true; rev = false
  kBT = []
  aves = []
  ses = []
  for χaux in χauxs
    kBT, ave, se = repcalcuβs(l=l, kmax=kmax, χ=χ, N=N, repnum=repnum, seqtrunc=seqtrunc, edgedim=χaux, saveplot=false, savedata=false, rev=rev)
    plot!(aveplot, kBT, ave, xlabel="kBT", ylabel="average of E/N", legend=:outerleft, label="χaux=$χaux", size=(800, 400))
    plot!(sesplot, kBT, se, xlabel="kBT", ylabel="standard error of E/N", legend=:outerleft, label="χaux=$χaux", size=(800, 400))
    if length(aves) == 0
      aves = deepcopy(kBT)
    end
    if length(ses) == 0
      ses = deepcopy(kBT)
    end
    aves = hcat(aves, ave)
    ses = hcat(ses, se)
  end
  infost = infos(l=l, N=N, χ=χ, χaux="-", kmax=kmax, seqtrunc=seqtrunc, rev=rev)
  savefig(aveplot, "cmp-ave-u-forχaux,$infost.png")
  savefig(sesplot, "cmp-se-u-forχaux,$infost.png")
  open("cmp-ave-u-forχaux,$infost.dat", "w") do fp
    Base.print_array(fp, aves)
  end
  open("cmp-se-u-forχaux,$infost.dat", "w") do fp
    Base.print_array(fp, ses)
  end
end

function cmpse_forN()
  aveplot = plot()
  sesplot = plot()
  Ns = [4, 8, 16, 32]
  l = 5; kmax = 500; χ = 10; repnum = 10; seqtrunc = true; rev = false; χaux = χ
  kBT = []
  aves = []
  ses = []
  for N in Ns
    kBT, ave, se = repcalcuβs(l=l, kmax=kmax, χ=χ, N=N, repnum=repnum, seqtrunc=seqtrunc, edgedim=χaux, saveplot=false, savedata=false, rev=rev)
    plot!(aveplot, kBT, ave, xlabel="kBT", ylabel="average of E/N", legend=:outerleft, label="N=$N", size=(800, 400))
    plot!(sesplot, kBT, se, xlabel="kBT", ylabel="standard error of E/N", legend=:outerleft, label="N=$N", size=(800, 400))
    if length(aves) == 0
      aves = deepcopy(kBT)
    end
    if length(ses) == 0
      ses = deepcopy(kBT)
    end
    aves = hcat(aves, ave)
    ses = hcat(ses, se)
  end
  infost = infos(l=l, N="-", χ=χ, χaux=χ, kmax=kmax, seqtrunc=seqtrunc, rev=rev)
  savefig(aveplot, "cmp-ave-u-forN,$infost.png")
  savefig(sesplot, "cmp-se-u-forN,$infost.png")
  open("cmp-ave-u-forN,$infost.dat", "w") do fp
    Base.print_array(fp, aves)
  end
  open("cmp-se-u-forN,$infost.dat", "w") do fp
    Base.print_array(fp, ses)
  end
end

@time cmpse_forχaux()
