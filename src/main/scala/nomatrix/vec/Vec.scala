package nomatrix.vec

import nomatrix.autograd.Value

/** ベクトルは「数の並び」。それ以上でもそれ以下でもない。 */
type Vec = Vector[Value]

object Vec:

  def fromDoubles(ds: Seq[Double]): Vec = ds.toVector.map(Value(_))

  def data(v: Vec): Vector[Double] = v.map(_.data)

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

  /** 全要素を同じ数倍する。 */
  def scale(v: Vec, k: Double): Vec = v.map(_ * k)

  def scale(v: Vec, k: Value): Vec = v.map(_ * k)

  /** 内積 = 対応する要素の積を全部足したもの。「似ている度合い」。 */
  def dot(a: Vec, b: Vec): Value =
    sameLength(a, b)
    Value.sum(a.zip(b).map(_ * _))

  /** 平均。 */
  def mean(v: Vec): Value = Value.sum(v) / v.length.toDouble

  /** 要素ごとの ReLU。 */
  def relu(v: Vec): Vec = v.map(_.relu)

  /** 連結。 */
  def concat(a: Vec, b: Vec): Vec = a ++ b

  /** softmax: 全要素を「合計 1 の割合」に変える。最大値を引いてから exp するのは桁あふれ対策。 */
  def softmax(v: Vec): Vec =
    val shift = v.map(_.data).max
    val exps = v.map(x => (x - shift).exp)
    val total = Value.sum(exps)
    exps.map(_ / total)

  /** LayerNorm: 平均 0・分散 1 に整える。 */
  def layerNorm(v: Vec, eps: Double = 1e-5): Vec =
    val m = mean(v)
    val centered = v.map(_ - m)
    val variance = mean(centered.map(c => c * c))
    val inv = (variance + eps).sqrt
    centered.map(_ / inv)
