package nomatrix.train

import scala.util.Random

/** 微分を使わない学習: 全パラメータにランダムなゆらぎを足して試し、良かったゆらぎの方へ寄る。
  *
  * パラメータは `FastTransformer.names` の順に並んだ数の配列。
  *
  * 基本形に加えて、速くするための 2 つの工夫を持つ（どちらも順伝播しか使わない）:
  *   - groups: パラメータをいくつかの組に分け、1 ステップに 1 組だけをゆらす（一段ずつ育てる）
  *   - lineSearch: 向きが決まったあと、歩幅を数通り試して一番良いものを採る
  */
object Evolution:

  final case class Settings(
      pairs: Int = 16,
      sigma: Double = 0.02,
      lr: Double = 0.01,
      useAdam: Boolean = true,
      ranks: Boolean = false,
      groups: Vector[Vector[Int]] = Vector.empty,
      lineSearch: Boolean = false,
      parallel: Boolean = false,
      rademacher: Boolean = false,
      orthogonal: Boolean = false,
      newton: Boolean = false
  )

  final case class State(step: Int, m: Array[Double], v: Array[Double])
  object State:
    def initial(size: Int): State = State(0, new Array[Double](size), new Array[Double](size))

  /** ゆらぎ同士を直交させる（Gram–Schmidt）。同じ方向を二度試す無駄が減る。長さは元のまま。 */
  private def orthogonalize(noises: Vector[Array[Double]]): Vector[Array[Double]] =
    def dot(a: Array[Double], b: Array[Double]): Double =
      var s = 0.0
      var i = 0
      while i < a.length do
        s += a(i) * b(i)
        i += 1
      s
    val out = Vector.newBuilder[Array[Double]]
    val done = scala.collection.mutable.ArrayBuffer.empty[(Array[Double], Double)] // (直交化済みのゆらぎ, その長さの二乗)
    for noise <- noises do
      val v = noise.clone()
      val originalNorm = math.sqrt(dot(v, v))
      for (u, squared) <- done do
        val p = dot(v, u) / squared
        var i = 0
        while i < v.length do
          v(i) -= p * u(i)
          i += 1
      val norm = math.sqrt(dot(v, v))
      val result = if norm < 1e-12 then noise else v.map(_ * originalNorm / norm)
      done += ((result, dot(result, result)))
      out += result
    out.result()

  /** 1 ステップ。`lossOf` は「このパラメータならどれだけ外すか」。返り値は (新パラメータ, 状態, 試したゆらぎの平均損失)。 */
  def step(params: Array[Double], lossOf: Array[Double] => Double, s: Settings, state: State, rng: Random): (Array[Double], State, Double) =
    val n = params.length
    // 今回ゆらす組。groups が無ければ全部
    val active = Array.fill(n)(s.groups.isEmpty)
    if s.groups.nonEmpty then s.groups(state.step % s.groups.length).foreach(i => active(i) = true)

    // ゆらぎ: 正規乱数か、±1（SPSA / Rademacher）
    val drawn = Vector.fill(s.pairs)(Array.tabulate(n) { i =>
      if !active(i) then 0.0 else if s.rademacher then (if rng.nextBoolean() then 1.0 else -1.0) else rng.nextGaussian()
    })
    val noises = if s.orthogonal then orthogonalize(drawn) else drawn
    val evaluate = (noise: Array[Double]) =>
      val plus = Array.tabulate(n)(i => params(i) + s.sigma * noise(i))
      val minus = Array.tabulate(n)(i => params(i) - s.sigma * noise(i))
      (lossOf(plus), lossOf(minus))
    // 評価は互いに独立なので、並列にできる（行列と同じ「同じ計算を同時にやる」）
    val scores: Vector[(Double, Double)] =
      if s.parallel then
        java.util.stream.IntStream.range(0, s.pairs).parallel().mapToObj(k => evaluate(noises(k))).toArray.toVector.map(_.asInstanceOf[(Double, Double)])
      else noises.map(evaluate)
    // 各ゆらぎの「+側と−側の差」。差が正ならゆらぎの逆向きが良い
    val raw = scores.map((lp, lm) => (lp - lm) / (2 * s.sigma))
    val weights =
      if s.ranks then
        val order = raw.indices.sortBy(raw(_))
        val w = new Array[Double](s.pairs)
        for (k, r) <- order.zipWithIndex do w(k) = (r.toDouble / (s.pairs - 1) - 0.5) * 2
        w.toVector
      else if s.newton then
        // 3 点 (+, 0, −) で方向ごとの曲率を測り、曲率が大きい方向は控えめに、小さい方向は大胆に
        val center = lossOf(params)
        val curvature = scores.map((lp, lm) => (lp - 2 * center + lm) / (s.sigma * s.sigma))
        val positive = curvature.filter(_ > 0)
        val floor = if positive.isEmpty then 1.0 else positive.sum / positive.length * 0.1
        raw.zip(curvature).map((g, c) => g / math.max(c, floor))
      else raw
    val direction = new Array[Double](n)
    for k <- 0 until s.pairs do
      val w = weights(k) / s.pairs
      val noise = noises(k)
      var i = 0
      while i < n do
        direction(i) += w * noise(i)
        i += 1
    val meanLoss = scores.map((a, b) => a + b).sum / (2 * s.pairs)

    // 向きから「1 単位の歩み」を作る（Adam なら歩幅を整えたもの）
    val t = state.step + 1
    val (unit, m, v) =
      if s.useAdam then
        val (b1, b2, eps) = (0.9, 0.999, 1e-8)
        val m = new Array[Double](n)
        val v = new Array[Double](n)
        val u = new Array[Double](n)
        var i = 0
        while i < n do
          m(i) = b1 * state.m(i) + (1 - b1) * direction(i)
          v(i) = b2 * state.v(i) + (1 - b2) * direction(i) * direction(i)
          val mHat = m(i) / (1 - math.pow(b1, t))
          val vHat = v(i) / (1 - math.pow(b2, t))
          u(i) = if active(i) then mHat / (math.sqrt(vHat) + eps) else 0.0
          i += 1
        (u, m, v)
      else (direction, state.m, state.v)

    def moved(lr: Double): Array[Double] = Array.tabulate(n)(i => params(i) - lr * unit(i))
    // 歩幅の探索: 半分・そのまま・倍 を試して一番良いものを採る
    val next =
      if s.lineSearch then
        Vector(0.5, 1.0, 2.0).map(f => moved(s.lr * f)).minBy(lossOf)
      else moved(s.lr)
    (next, State(t, m, v), meanLoss)
