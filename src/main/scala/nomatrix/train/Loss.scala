package nomatrix.train

import nomatrix.nn.Tokens
import nomatrix.vec.Vec

object Loss:

  /** 交差エントロピー = -log(正解の確率) = logSumExp(logits) - logits(正解)。 */
  def crossEntropy(logits: Vec, target: Int): Double =
    Vec.logSumExp(logits) - logits(target)

  /** 列全体の損失は各位置の平均。 */
  def sequence(logits: Tokens, targets: Vector[Int]): Double =
    require(logits.length == targets.length, "ロジットと正解の長さが違う")
    logits.zip(targets).map(crossEntropy).sum / targets.length
