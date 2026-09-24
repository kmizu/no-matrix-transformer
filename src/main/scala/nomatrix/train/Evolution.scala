package nomatrix.train

import scala.util.Random

/** 微分を使わない学習: 全パラメータにランダムなゆらぎを足して試し、良かったゆらぎの方へ寄る。
  *
  * パラメータは `FastTransformer.names` の順に並んだ数の配列。
  */
object Evolution:

  final case class Settings(pairs: Int = 16, sigma: Double = 0.02, lr: Double = 0.01, useAdam: Boolean = true, ranks: Boolean = false)

  final case class State(step: Int, m: Array[Double], v: Array[Double])
  object State:
    def initial(size: Int): State = State(0, new Array[Double](size), new Array[Double](size))

  /** 1 ステップ。`lossOf` は「このパラメータならどれだけ外すか」。返り値は (新パラメータ, 状態, 試したゆらぎの平均損失)。 */
  def step(params: Array[Double], lossOf: Array[Double] => Double, s: Settings, state: State, rng: Random): (Array[Double], State, Double) =
    val n = params.length
    val noises = Array.fill(s.pairs)(Array.fill(n)(rng.nextGaussian()))
    val scores = noises.map { noise =>
      val plus = Array.tabulate(n)(i => params(i) + s.sigma * noise(i))
      val minus = Array.tabulate(n)(i => params(i) - s.sigma * noise(i))
      (lossOf(plus), lossOf(minus))
    }
    // 各ゆらぎの「+側と−側の差」。差が正ならゆらぎの逆向きが良い
    val raw = scores.map((lp, lm) => (lp - lm) / (2 * s.sigma))
    val weights =
      if s.ranks then
        // 大きさは捨てて順位だけ使う（外れ値に強い）
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
    if s.useAdam then
      val t = state.step + 1
      val (b1, b2, eps) = (0.9, 0.999, 1e-8)
      val m = new Array[Double](n)
      val v = new Array[Double](n)
      val next = new Array[Double](n)
      var i = 0
      while i < n do
        m(i) = b1 * state.m(i) + (1 - b1) * direction(i)
        v(i) = b2 * state.v(i) + (1 - b2) * direction(i) * direction(i)
        val mHat = m(i) / (1 - math.pow(b1, t))
        val vHat = v(i) / (1 - math.pow(b2, t))
        next(i) = params(i) - s.lr * mHat / (math.sqrt(vHat) + eps)
        i += 1
      (next, State(t, m, v), meanLoss)
    else
      (Array.tabulate(n)(i => params(i) - s.lr * direction(i)), state.copy(step = state.step + 1), meanLoss)
