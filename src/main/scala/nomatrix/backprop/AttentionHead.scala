package nomatrix.backprop

import nomatrix.nn.Params


import scala.util.Random

/** トークンごとのベクトルの列。文（トークン列）を通した後の「各トークンの状態」。 */
type Tokens = Vector[GVec]

/** 注意（attention）ひとつ分。
  *
  *   1. 各トークンから query / key / value の 3 つの数の並びを作る。
  *   2. トークン i は、自分より前（と自分）のトークン j との「似ている度合い」を query_i と key_j の内積で測る。
  *   3. 似ている度合いを softmax で割合にして、その割合で value_j を混ぜ合わせる。
  *
  * 未来のトークン（j > i）は見ない。これが「因果的」注意。
  */
final case class AttentionHead(query: Dense, key: Dense, value: Dense, headDim: Int):

  def apply(xs: Tokens): Tokens =
    val qs = xs.map(query(_))
    val ks = xs.map(key(_))
    val vs = xs.map(value(_))
    val scale = 1.0 / math.sqrt(headDim.toDouble)
    xs.indices.toVector.map { i =>
      val scores = (0 to i).toVector.map(j => GVec.dot(qs(i), ks(j)) * scale)
      val weights = GVec.softmax(scores)
      (0 to i).map(j => GVec.scale(vs(j), weights(j))).reduce(GVec.add)
    }

object AttentionHead:

  def init(prefix: String, dModel: Int, headDim: Int, rng: Random): Params =
    Dense.init(s"$prefix.query", dModel, headDim, rng) ++
      Dense.init(s"$prefix.key", dModel, headDim, rng) ++
      Dense.init(s"$prefix.value", dModel, headDim, rng)

  def load(p: ParamValues, prefix: String, dModel: Int, headDim: Int): AttentionHead =
    AttentionHead(
      Dense.load(p, s"$prefix.query", dModel, headDim),
      Dense.load(p, s"$prefix.key", dModel, headDim),
      Dense.load(p, s"$prefix.value", dModel, headDim),
      headDim
    )
