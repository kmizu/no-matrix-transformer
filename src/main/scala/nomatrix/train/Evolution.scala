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
      lineSearch: Boolean = false
  )

  final case class State(step: Int, m: Array[Double], v: Array[Double])
  object State:
    def initial(size: Int): State = State(0, new Array[Double](size), new Array[Double](size))

  /** 1 ステップ。`lossOf` は「このパラメータならどれだけ外すか」。返り値は (新パラメータ, 状態, 試したゆらぎの平均損失)。 */
  def step(params: Array[Double], lossOf: Array[Double] => Double, s: Settings, state: State, rng: Random): (Array[Double], State, Double) =
    val n = params.length
    // 今回ゆらす組。groups が無ければ全部
    val active = Array.fill(n)(s.groups.isEmpty)
    if s.groups.nonEmpty then s.groups(state.step % s.groups.length).foreach(i => active(i) = true)

    val noises = Array.fill(s.pairs)(Array.tabulate(n)(i => if active(i) then rng.nextGaussian() else 0.0))
    val scores = noises.map { noise =>
      val plus = Array.tabulate(n)(i => params(i) + s.sigma * noise(i))
      val minus = Array.tabulate(n)(i => params(i) - s.sigma * noise(i))
      (lossOf(plus), lossOf(minus))
    }
    // 各ゆらぎの「+側と−側の差」。差が正ならゆらぎの逆向きが良い
    val raw = scores.map((lp, lm) => (lp - lm) / (2 * s.sigma))
    val weights =
      if s.ranks then
        val order = raw.indices.sortBy(raw(_))
        val w = new Array[Double](s.pairs)
        for (k, r) <- order.zipWithIndex do w(k) = (r.toDouble / (s.pairs - 1) - 0.5) * 2
        w
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
