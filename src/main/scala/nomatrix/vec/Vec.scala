package nomatrix.vec

/** ベクトルは「数の並び」。それ以上でもそれ以下でもない。 */
type Vec = Vector[Double]

object Vec:

  private def sameLength(a: Vec, b: Vec): Unit =
    require(a.length == b.length, s"長さが違う: ${a.length} と ${b.length}")

  /** 要素ごとの足し算。 */
  def add(a: Vec, b: Vec): Vec =
    sameLength(a, b)
    a.zip(b).map(_ + _)

  /** 要素ごとの引き算。 */
  def sub(a: Vec, b: Vec): Vec =
    sameLength(a, b)
    a.zip(b).map(_ - _)

  /** 要素ごとの掛け算。 */
  def mul(a: Vec, b: Vec): Vec =
    sameLength(a, b)
    a.zip(b).map(_ * _)

  /** 全要素を同じ数倍する。 */
  def scale(v: Vec, k: Double): Vec = v.map(_ * k)

  /** 内積 = 対応する要素の積を全部足したもの。「似ている度合い」。 */
  def dot(a: Vec, b: Vec): Double =
    sameLength(a, b)
    a.zip(b).map(_ * _).sum

  /** 平均。 */
  def mean(v: Vec): Double = v.sum / v.length

  /** 要素ごとの ReLU（負なら 0）。 */
  def relu(v: Vec): Vec = v.map(x => math.max(0.0, x))

  /** 連結。 */
  def concat(a: Vec, b: Vec): Vec = a ++ b

  /** softmax: 全要素を「合計 1 の割合」に変える。最大値を引いてから exp するのは桁あふれ対策。 */
  def softmax(v: Vec): Vec =
    val shift = v.max
    val exps = v.map(x => math.exp(x - shift))
    val total = exps.sum
    exps.map(_ / total)

  /** log(Σ exp(x_i))。最大値を引いてから計算すると桁あふれしない。 */
  def logSumExp(v: Vec): Double =
    val shift = v.max
    math.log(v.map(x => math.exp(x - shift)).sum) + shift

  /** LayerNorm: 平均 0・分散 1 に整える。 */
  def layerNorm(v: Vec, eps: Double = 1e-5): Vec =
    val m = mean(v)
    val centered = v.map(_ - m)
    val variance = mean(centered.map(c => c * c))
    val inv = 1.0 / math.sqrt(variance + eps)
    centered.map(_ * inv)
