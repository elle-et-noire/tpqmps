using ITensors
using LinearAlgebra
using Plots
using Base.Threads

function sample01()
  i = Index(2)
  j = Index(2)

  ψ = ITensor(i)
  H = ITensor(i, j)
  ψ2 = ITensor(j)

  ψ[i=>2] = 1
  H[i=>1, j=>1] = 1
  H[i=>2, j=>2] = -1
  ψ2[j=>2] = 1

  print(H)
  print(ψ*H*ψ2)
end

function qr01()
  i = Index(3)
  j = Index(3)
  k = Index(3)
  T = randomITensor(i, j, k)
  Q, R = qr(T, (i, k))
  println(Q * R ≈ T)
  q = commoninds(Q, R)
  println(Q)
  println(R)
end

function svd01()
  i = Index(3)
  j = Index(3)
  m = Index(3)
  k = Index(3)
  W = randomITensor(i, j, m, k)
  U, S, V = svd(W, (j, i))
  println(U * S * V ≈ W)
  println(S)
end

function transverseIsing()
  N = 3
  # mpobonds = [Index(3) for i in 1:N-1]
  mpotops = siteinds(2, N)
  mpobottoms = siteinds(2, N)
  mpobonds = siteinds(3, N - 1)
  lhoptemplate = zeros(2, 3, 2)
  rhoptemplate = zeros(2, 3, 2)
  lhoptemplate[:, 1, :] = rhoptemplate[:, 3, :] = I(2)
  lhoptemplate[:, 2, :] = rhoptemplate[:, 2, :] = I(2)
  lhop = ITensor(lhoptemplate, mpotops[1], mpobonds[1], mpobottoms[1])
  rhop = ITensor(rhoptemplate, mpotops[end], mpobonds[end], mpobottoms[end])
  mhoptemplate = zeros(3, 2, 3, 2)
  mhoptemplate[1, :, 1, :] = I(2)
  mhoptemplate[3, :, 3, :] = I(2)
  mhoptemplate[1, :, 2, :] = mhoptemplate[2, :, 3, :] = [1 0:0 -1]
  mhop = [ITensor(mhoptemplate, mpobonds[i-1], mpotops[i], mpobonds[i], mpobottoms[i]) for i in 2:N-1]
end

function tomps()
  i = Index(2)
  j = Index(2)
  k = Index(2)
  l = Index(2)
  m = Index(2)
  cutoff = 1E-8
  maxdim = 10
  T = randomITensor(i,j,k,l,m)
  M = MPS(T,(i,j,k,l,m);cutoff=cutoff,maxdim=maxdim)
  println(M)
  println(M[1])
  println(M[2])
end

function expect01()
  N = 10
  chi = 4
  sites = siteinds("S=1/2",N)
  psi = randomMPS(sites,chi)
  magz = expect(psi,"Sz")
  for (j,mz) in enumerate(magz)
      println("$j $mz")
  end
  zzcorr = correlation_matrix(psi,"Sz","Sz")
  for i in 1:N-1
    println(zzcorr[i, i + 1])
  end
end

function schmidt01()
  d = 2
  N = 3
  Ψ = rand(d^N)

  Ψ1_23 = reshape(Ψ, (d, d))

end

function boxmuller()
  x = rand()
  y = rand()
  return sqrt(-2 * log(x)) * exp(2pi * im * y)
end

function reshapeunitary()
  χ = 3
  d = 2
  q, _ = reshape([boxmuller() for i in 1:(χ*d)^2], (χ * d, χ * d)) |> qr
  println(q * q' ≈ I)
  u = reshape(q, (d, χ, d, χ))

  ret = zeros(χ, χ)
  for i in 1:d
    for α in 1:χ
      for β2 in 1:χ
        for β in 1:χ
          ret[β2, β] += real(u[i, α, 1, β2] * conj(u[i, α, 1, β]))
        end
      end
    end
  end
  display(ret)
end

function speedtest_fast()
  i, j, k = siteinds(2, 6), siteinds(2, 6), siteinds(2, 6)
  A = randomITensor(i, j)
  B = randomITensor(j, k)
  C = randomITensor(k)
  @time A * (B * C)
end

function speedtest_slow()
  i, j, k = siteinds(2, 6), siteinds(2, 6), siteinds(2, 6)
  A = randomITensor(i, j)
  B = randomITensor(j, k)
  C = randomITensor(k)
  @time (A * B) * C
end

function optimization_quest()
  speedtest_fast()
  speedtest_slow()
  speedtest_fast()
  speedtest_slow()
end

function entanglemententropytest()
  i, j, k, l = siteinds(3, 4)
  a, b, c = siteinds(2, 3)
  A = randomITensor(i, j, a)
  B = randomITensor(j, k, b)
  C = randomITensor(k, l, c)
  bulk = A * B * C
  norm2_exact = (bulk * conj(bulk))[]
  println("norm2_exact=$norm2_exact")
  A /= norm2_exact^inv(6)
  B /= norm2_exact^inv(6)
  C /= norm2_exact^inv(6)
  bulk /= sqrt(norm2_exact)
  U, S, V = svd(bulk, (i, a))
  ee_a_bc_exact = -sum(storage(S) .|> x -> x^2 * log(x^2))
  println("ee_a_bc_exact=$ee_a_bc_exact")
  U, S, V = svd(bulk, a)
  ee_a_bc_exact_sub = -sum(storage(S) .|> x -> x^2 * log(x^2))
  println("ee_a_bc_exact_sub=$ee_a_bc_exact_sub")
  U, S, V = svd(bulk, (l, c))
  ee_ab_c_exact = -sum(storage(S) .|> x -> x^2 * log(x^2))
  println("ee_ab_c_exact=$ee_ab_c_exact")
  U, S, V = svd(bulk, b)
  ee_ac_b_exact = -sum(storage(S) .|> x -> x^2 * log(x^2))
  println("ee_ac_b_exact=$ee_ac_b_exact")
  auxL, S, V = svd(A, i)
  Acanl = S * V
  U, S, V = svd(Acanl * B, (commonind(auxL, Acanl), a))
  Acanl = U
  Bcanl = S * V
  U, S, V = svd(Bcanl * C, (commonind(Acanl, Bcanl), b))
  Bcanl = U
  Ccanl = S * V
  ee_ab_c_leftcan = -sum(storage(S) .|> x -> x^2 * log(x^2))
  println("ee_ab_c_leftcan=$ee_ab_c_leftcan")

  auxR, S, V = svd(Ccanl, l)
  Ccan = S * V
  U, Sbc, V = svd(Bcanl * Ccan, (commonind(auxR, Ccan), c))
  Ccan = U
  Bcan = V * Sbc
  ee_ab_c_can = -sum(storage(Sbc) .|> x -> x^2 * log(x^2))
  println("ee_ab_c_can=$ee_ab_c_can")
  U, Sab, V = svd(Acanl * Bcan, (commonind(Ccan, Bcan), b))
  Bcan = U
  Acan = V * Sab
  ee_a_bc_can = -sum(storage(Sab) .|> x -> x^2 * log(x^2))
  println("ee_a_bc_can=$ee_a_bc_can")
  U, S, V = svd(Acan * Bcan, (commonind(Acan, auxL), a))
  Acan = U
  Bcan = S * V
  U, S, V = svd(Bcan, b)
  ee_ac_b_can = -sum(storage(S) .|> x -> x^2 * log(x^2))
  println("ee_ac_b_can=$ee_ac_b_can")
end

# λ : array of singular values(not necessarily normalized)
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

function Ψcenterentropies2(_Ψ, _Ψbonds)
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
  # for (i, p) in enumerate(Ψ)
  #   println("\n\n$i : ")
  #   display(p)
  # end

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
    println("\n")
    λ = eigvals(mbulk |> matrix) |> real
    entropies[count+1] = -sum(λ ./ sum(λ) .|> x -> abs(x) * log(abs(x)))
  end

  return entropies
end

function Ψcenterentropies(_Ψ, _Ψbonds, _physinds)
  N = length(_Ψ)
  entropies = Vector{Float64}(undef, N ÷ 2)
  Ψ = deepcopy(_Ψ)
  Ψbonds = deepcopy(_Ψbonds)
  χ = dim(Ψbonds[end])
  physinds = deepcopy(_physinds)
  # add aux explicitly to process uniformly
  push!(physinds, Ψbonds[end])
  pushfirst!(physinds, Ψbonds[1])

  tmpb = Index(dim(Ψbonds[end]))
  replaceind!(Ψ[end], Ψbonds[end] => tmpb)
  Ψbonds[end] = tmpb
  tmpb = Index(dim(Ψbonds[1]))
  replaceind!(Ψ[1], Ψbonds[1] => tmpb)
  Ψbonds[1] = tmpb
  push!(Ψbonds, Index(1))
  pushfirst!(Ψbonds, Index(1))
  push!(Ψ, ITensor(I(χ), Ψbonds[end-1], Ψbonds[end], physinds[end]))
  pushfirst!(Ψ, ITensor(I(χ), Ψbonds[1], Ψbonds[2], physinds[1]))

  println("prepare complete")
  # println("\n\n ----- initial state -----")
  # display(physinds)

  # put two sites together from center and move to end(right edge) and put two sites together from center of rest... and so on
  for count in 0:N÷2-1
    # center position(left of pair)
    cpos = N ÷ 2 + 1 - count
    while cpos < N + 1 - 2count
      A, B, _, Ψbonds[cpos+2] = factorize(Ψ[cpos+1] * Ψ[cpos+2], (physinds[cpos+1], Ψbonds[cpos+3]))
      Ψ[cpos+2] = A
      Ψ[cpos+1] = B # B has physinds[cpos+2]
      physinds[cpos+2], physinds[cpos+1] = physinds[cpos+1], physinds[cpos+2]
      A, B, _, Ψbonds[cpos+1] = factorize(Ψ[cpos] * Ψ[cpos+1], (physinds[cpos], Ψbonds[cpos+2]))
      Ψ[cpos+1] = A
      Ψ[cpos] = B # B has physinds[cpos+2]
      physinds[cpos+1], physinds[cpos] = physinds[cpos], physinds[cpos+1]
      cpos += 1
    end
    # println("\n\n ----- count = $count -----")
    # display(physinds)
  end

  println("move complete.")
  # for (i, p) in enumerate(Ψ)
  #   println("\n\n$i:")
  #   display(p)
  # end

  # println("\n$(foldl(*, Ψ) * δ(Ψbonds[1], Ψbonds[end]) ≈ foldl(*, _Ψ))")

  for site in 1:length(Ψ)-1
    U, S, V = svd(Ψ[site] * Ψ[site+1], uniqueinds(Ψ[site], Ψ[site+1]))
    Ψ[site] = U
    Ψ[site+1] = S * V
  end
  println("right canonical.")
  for site in length(Ψ)-1:-1:2
    U, S, V = svd(Ψ[site] * Ψ[site+1], uniqueinds(Ψ[site+1], Ψ[site]))
    Ψ[site+1] = U
    Ψ[site] = V * S
    if site & 1 == 0
      entropies[(N-site)÷2+1] = sings2ent(storage(S))
    end
  end
  println("center entropies complete.")

  return entropies
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

function entropytest2()
  N = 8
  # Ψ, Ψbonds, physinds = genΨ(sitenum = N, physdim = 2, bonddim = 3, rightunitary=true)
  Ψ, Ψbonds, physinds = genΨnocanonical(sitenum=N, physdim=2, bonddim=6)
  bulk = foldl(*, Ψ)

  # exactentropies = zeros(Float64, N + 1)
  # _, S, _ = svd(bulk, Ψbonds[1])
  # exactentropies[1] = entropy(storage(S))
  # for site in 1:N-1
  #   _, S, _ = svd(bulk, (Ψbonds[1], physinds[1:site]...))
  #   exactentropies[site+1] = entropy(storage(S))
  # end
  # _, S, _ = svd(bulk, Ψbonds[end])
  # exactentropies[end] = entropy(storage(S))

  # canentropies = Ψentropies(Ψ, Ψbonds)

  # plot([1:N+1;], exactentropies, xlabel="i", ylabel="S_i", label="exact")
  # plot!([1:N+1;], canentropies, label="canonical")
  # savefig("entropytest2-edge.png")

  println("uouo")
  exactcenterentropies = zeros(Float64, N ÷ 2)
  for n in 2:2:N
    println("guee")
    _, S, _ = svd(bulk, (physinds[(N-n)÷2+1:(N+n)÷2]...))
    # display(storage(S))
    println("\n")
    exactcenterentropies[n÷2] = sings2ent(storage(S))
  end
  println("yy")
  @time cancenterentropies = Ψcenterentropies(Ψ, Ψbonds, physinds)
  @time cantrunccenterents = Ψcenterentropies2(Ψ, Ψbonds)
  println("draw!")

  plot([2:2:N;], exactcenterentropies, xlabel="n", ylabel="S_n^c", label="exact")
  plot!([2:2:N;], cancenterentropies, label="canonical")
  plot!([2:2:N;], cantrunccenterents, label="power-diagonalize")
  savefig("entropytest2-center.png")
  println("\n\nveryexactentropies:")
  display(exactcenterentropies)
  println("\n")
  display(cancenterentropies)
  display(cantrunccenterents)
end

function _writeΨ(Ψ, filename)
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

function readwritetensor()
  i = siteinds(40, 4)
  a = siteinds(2, 3)
  A = randomITensor(i[1], a[1], i[2])
  B = randomITensor(i[2], a[2], i[3])
  C = randomITensor(i[3], a[3], i[4])
  Ψ = [A, B, C]

  _writeΨ(Ψ, "readwritetensorsample.txt")

  Ψ2 = []
  current = Vector{Float64}()
  num = 1

  # open("readwritetensorsample.txt", "r") do fp
  for line in eachline("readwritetensorsample.txt")
    if line == ""
      push!(Ψ2, ITensor(current, i[num], a[num], i[num+1]))
      current = Vector{ComplexF64}()
      num += 1
    else
      push!(current, parse(ComplexF64, line))
    end
  end

  for (p1, p2) in zip(Ψ, Ψ2)
    println(storage(p1) ≈ storage(p2))
  end
end

function compfor()
  [(x, y) for x in 1:3 for y in 1:3] # direct product
  [(x, y) for x in 1:3, y in 1:3] # matrix
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

function Ψentropies2(_Ψ, Ψbonds)
  Ψ = deepcopy(_Ψ)
  N = length(Ψ)
  lefttask = Threads.@spawn for site in 1:N÷2-1
    Q, R = qr(Ψ[site] * Ψ[site+1], uniqueinds(Ψ[site], Ψ[site+1]))
    Ψ[site] = Q
    Ψ[site+1] = R
  end
  righttask = Threads.@spawn for site in 1:N÷2-1
    Q, R = qr(Ψ[N+1-site] * Ψ[N-site], uniqueinds(Ψ[N+1-site], Ψ[N-site]))
    Ψ[N+1-site] = Q
    Ψ[N-site] = R
  end
  fetch(lefttask)
  fetch(righttask)

  Ucent, Scent, Vcent = svd(Ψ[N÷2] * Ψ[N÷2+1], uniqueinds(Ψ[N÷2], Ψ[N÷2+1]))

  ret = zeros(Float64, N + 1)
  ret[N÷2+1] = sings2ent(Scent |> storage)

  lefttask = Threads.@spawn begin
    Ψ[N÷2] = Ucent * Scent
    for site in N÷2-1:-1:1
      U, S, V = svd(Ψ[site] * Ψ[site+1], uniqueinds(Ψ[site+1], Ψ[site]))
      Ψ[site] = S * V
      Ψ[site+1] = U
      ret[site+1] = sings2ent(S |> storage)
    end
    _, S, _ = svd(Ψ[1], Ψbonds[1])
    ret[1] = sings2ent(S |> storage)
  end
  righttask = Threads.@spawn begin
    Ψ[1+N÷2] = Scent * Vcent
    for site in 1+N÷2:N-1
      U, S, V = svd(Ψ[site] * Ψ[site+1], uniqueinds(Ψ[site], Ψ[site+1]))
      Ψ[site] = U
      Ψ[site+1] = S * V
      ret[site+1] = sings2ent(S |> storage)
    end
    _, S, _ = svd(Ψ[end], Ψbonds[end])
    ret[end] = sings2ent(S |> storage)
  end
  fetch(lefttask)
  fetch(righttask)
  return ret
end

function cantest()
  N = 8
  χ = 20
  Ψ, Ψbonds, physinds = genΨnocanonical(sitenum=N, physdim=2, bonddim=χ, withaux=false)
  Ψ /= norm2(Ψ, Ψbonds)^inv(2N)

  Ψ2 = deepcopy(Ψ)
  Ψ2bonds = deepcopy(Ψbonds)
  Ψ2bonds[1] = Ψbonds[1]
  Ψ2bonds[end] = Ψbonds[end]
  for site in 1:N÷2-1
    Q, R = qr(Ψ[site] * Ψ[site+1], uniqueinds(Ψ[site], Ψ[site+1]))
    Ψ2[site] = Q
    Ψ2[site+1] = R
    Ψ2bonds[site+1] = commonind(Q, R)

    Q, R = qr(Ψ[N+1-site] * Ψ[N-site], uniqueinds(Ψ[N+1-site], Ψ[N-site]))
    Ψ2[N+1-site] = Q
    Ψ2[N-site] = R
    Ψ2bonds[N+1-site] = commonind(Q, R)
  end
  Ψ2bonds[N÷2+1] = commonind(Ψ2[N÷2], Ψ2[N÷2+1])
  Ucent, Scent, Vcent = svd(Ψ2[N÷2] * Ψ2[N÷2+1], uniqueinds(Ψ2[N÷2], Ψ2[N÷2+1]))
  # display(Scent |> storage)
  # println("\n----- comparison -----")
  # println(sum(storage(Scent).^2))
  # println(norm2(Ψ2, Ψ2bonds))
  sqrtScent = ITensor(sqrt.(Scent |> matrix), inds(Scent))

  entstry = zeros(Float64, N - 1)
  entstry[N÷2] = sings2ent(Scent |> storage)
  Ψhalv = deepcopy(Ψ2)
  Ψhalv[N÷2] = Ucent * sqrtScent
  Ψhalvbonds = similar(Ψ2bonds)
  Ψhalvbonds[1] = Ψ2bonds[1]
  Ψhalvbonds[end] = Ψ2bonds[end]
  # Ψhalvbonds[N÷2+1] = Ψ2bonds[N÷2+1]
  Ψhalvbonds[N÷2+1] = commonind(Scent, Vcent)
  for site in N÷2-1:-1:1
    U, S, V = svd(Ψhalv[site] * Ψhalv[site+1], uniqueinds(Ψhalv[site+1], Ψhalv[site]))
    Ψhalv[site] = S * V
    Ψhalv[site+1] = U
    entstry[site] = sings2ent(S |> storage)
    Ψhalvbonds[site+1] = commonind(U, S)
  end

  Ψhalv[1+N÷2] = sqrtScent * Vcent
  replaceind!(Ψhalv[1+N÷2], commonind(Ucent, Scent) => Ψhalvbonds[N÷2+1])
  for site in 1+N÷2:N-1
    U, S, V = svd(Ψhalv[site] * Ψhalv[site+1], uniqueinds(Ψhalv[site], Ψhalv[site+1]))
    Ψhalv[site] = U
    Ψhalv[site+1] = S * V
    entstry[site] = sings2ent(S |> storage)
    Ψhalvbonds[site+1] = commonind(U, S)
  end

  entΨ1 = Ψentropies(Ψ2, Ψ2bonds)
  entΨhalv = Ψentropies(Ψhalv, Ψhalvbonds)
  plot([1:N-1;], entstry, xlabel="bond", ylabel="entropy", label="entstry")
  plot!([0:N;], entΨ1, xlabel="bond", ylabel="entropy", label="entΨ1")
  plot!([0:N;], entΨhalv, xlabel="bond", ylabel="entropy", label="entΨhalv")
  savefig("entropytest.png")
  println("\n entstry:")
  display(entstry)
  println("\n\n entΨ1:")
  display(entΨ1)
  println("\n\n entΨhalv:")
  display(entΨhalv)
end

function cantest2()
  N = 64
  χ = 40
  Ψ, Ψbonds, physinds = genΨnocanonical(sitenum=N, physdim=2, bonddim=χ, withaux=true)
  Ψ /= norm2(Ψ, Ψbonds)^inv(2N)

  uo = Ψentropies(Ψ, Ψbonds)
  gue = Ψentropies2(Ψ, Ψbonds)

  @time Ψentropies(Ψ, Ψbonds)
  @time Ψentropies2(Ψ, Ψbonds)

  plot([0:N;], gue, xlabel="bond", ylabel="entropy", label="Ψent2", legend = :outerleft)
  plot!([0:N;], uo, xlabel="bond", ylabel="entropy", label="Ψent", legend = :outerleft)
  savefig("entropytest2.png")
end

# function writeΨ()

# end

# entropytest2()
cantest2()