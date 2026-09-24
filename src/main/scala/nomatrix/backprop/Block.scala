package nomatrix.backprop

import nomatrix.nn.Params


import scala.util.Random

/** Transformer ブロック。
  *
  *   x  = x + MultiHead(LayerNorm(x))   … トークン同士で情報を混ぜる
  *   x  = x + FeedForward(LayerNorm(x)) … トークンごとに考える
  *
  * 「x +」が残差接続。元の情報を捨てずに、差分だけを学ぶ。
  */
final case class Block(norm1: LayerNorm, attention: MultiHead, norm2: LayerNorm, feedForward: FeedForward):

  def apply(xs: Tokens): Tokens =
    val mixed = attention(xs.map(norm1(_)))
    val afterAttention = xs.zip(mixed).map(GVec.add)
    afterAttention.map(x => GVec.add(x, feedForward(norm2(x))))

object Block:

  def init(prefix: String, dModel: Int, heads: Int, hidden: Int, rng: Random): Params =
    LayerNorm.init(s"$prefix.norm1", dModel) ++
      MultiHead.init(s"$prefix.attn", dModel, heads, rng) ++
      LayerNorm.init(s"$prefix.norm2", dModel) ++
      FeedForward.init(s"$prefix.ff", dModel, hidden, rng)

  def load(p: ParamValues, prefix: String, dModel: Int, heads: Int, hidden: Int): Block =
    Block(
      LayerNorm.load(p, s"$prefix.norm1", dModel),
      MultiHead.load(p, s"$prefix.attn", dModel, heads),
      LayerNorm.load(p, s"$prefix.norm2", dModel),
      FeedForward.load(p, s"$prefix.ff", dModel, hidden)
    )
