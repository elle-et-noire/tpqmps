using ITensors
using LinearAlgebra

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

function entanglemententropy()
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
  # V * conj(prime(V, commonind(V, Sab))) |> matrix |> display
  # comb = combiner(commonind(auxL, V), commonind(V, Sab))
  # V2 = V * comb
  # V2 * conj(prime(V2, combinedind(comb))) |> matrix |> display
  # [-2(sab*sbc)^2 * log(sab*sbc) for (sab, sbc) in zip(storage(Sab), storage(Sbc))] |> sum |> println
  U, S, V = svd(Acan * Bcan, (commonind(Acan, auxL), a))
  Acan = U
  Bcan = S * V
  U, S, V = svd(Bcan, b)
  ee_ac_b_can = -sum(storage(S) .|> x -> x^2 * log(x^2))
  println("ee_ac_b_can=$ee_ac_b_can")
end

entanglemententropy()