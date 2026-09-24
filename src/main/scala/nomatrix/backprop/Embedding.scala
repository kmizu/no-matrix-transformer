package nomatrix.backprop

import nomatrix.nn.Params


import scala.util.Random

/** 埋め込み = 「番号 → 数の並び」の引き当て表。トークンにも位置にも使う。 */
final case class Embedding(table: Vector[GVec]):
  def apply(id: Int): GVec = table(id)

object Embedding:

  private def name(prefix: String, t: Int, d: Int) = s"$prefix.t$t.d$d"

  def init(prefix: String, count: Int, dim: Int, rng: Random): Params =
    val entries =
      for t <- 0 until count; d <- 0 until dim
      yield name(prefix, t, d) -> rng.nextGaussian() * 0.1
    Params(entries.toMap)

  def load(p: ParamValues, prefix: String, count: Int, dim: Int): Embedding =
    Embedding((0 until count).toVector.map(t => (0 until dim).toVector.map(d => p(name(prefix, t, d)))))
