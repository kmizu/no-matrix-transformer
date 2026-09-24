package nomatrix.nn

import nomatrix.vec.Vec
import scala.util.Random

/** 複数の注意ヘッドを並べ、それぞれの答えを連結して、Dense で dModel 個の数に戻す。 */
final case class MultiHead(heads: Vector[AttentionHead], output: Dense):

  def apply(xs: Tokens): Tokens =
    val perHead = heads.map(_(xs)) // heads × tokens
    xs.indices.toVector.map { i =>
      val joined = perHead.map(_(i)).reduce(Vec.concat)
      output(joined)
    }

object MultiHead:

  private def check(dModel: Int, heads: Int): Int =
    require(dModel % heads == 0, s"dModel=$dModel はヘッド数 $heads で割り切れない")
    dModel / heads

  def init(prefix: String, dModel: Int, heads: Int, rng: Random): Params =
    val headDim = check(dModel, heads)
    val headParams = (0 until heads).map(h => AttentionHead.init(s"$prefix.head$h", dModel, headDim, rng))
    headParams.foldLeft(Params.empty)(_ ++ _) ++ Dense.init(s"$prefix.output", dModel, dModel, rng)

  def load(p: ParamValues, prefix: String, dModel: Int, heads: Int): MultiHead =
    val headDim = check(dModel, heads)
    MultiHead(
      (0 until heads).toVector.map(h => AttentionHead.load(p, s"$prefix.head$h", dModel, headDim)),
      Dense.load(p, s"$prefix.output", dModel, dModel)
    )
