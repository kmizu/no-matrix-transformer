package nomatrix.backprop

import nomatrix.nn.Params



/** LayerNorm = 数の並びを平均 0・分散 1 に整えてから、要素ごとにゲインとバイアスをかける。 */
final case class LayerNorm(gain: GVec, bias: GVec):
  def apply(x: GVec): GVec = GVec.add(GVec.mul(gain, GVec.layerNorm(x)), bias)

object LayerNorm:

  private def gainName(prefix: String, d: Int) = s"$prefix.g$d"
  private def biasName(prefix: String, d: Int) = s"$prefix.b$d"

  /** ゲイン 1・バイアス 0 から始める（＝最初は素の layerNorm）。 */
  def init(prefix: String, dim: Int): Params =
    val gains = (0 until dim).map(d => gainName(prefix, d) -> 1.0)
    val biases = (0 until dim).map(d => biasName(prefix, d) -> 0.0)
    Params((gains ++ biases).toMap)

  def load(p: ParamValues, prefix: String, dim: Int): LayerNorm =
    LayerNorm(
      (0 until dim).toVector.map(d => p(gainName(prefix, d))),
      (0 until dim).toVector.map(d => p(biasName(prefix, d)))
    )
