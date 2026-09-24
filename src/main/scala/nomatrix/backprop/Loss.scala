package nomatrix.backprop





object Loss:

  /** log(Σ exp(x_i))。最大値を引いてから計算すると桁あふれしない。 */
  def logSumExp(xs: GVec): Value =
    val shift = xs.map(_.data).max
    Value.sum(xs.map(x => (x - shift).exp)).log + shift

  /** 交差エントロピー = -log(正解の確率) = logSumExp(logits) - logits(正解)。 */
  def crossEntropy(logits: GVec, target: Int): Value =
    logSumExp(logits) - logits(target)

  /** 列全体の損失は各位置の平均。 */
  def sequence(logits: Tokens, targets: Vector[Int]): Value =
    require(logits.length == targets.length, "ロジットと正解の長さが違う")
    val losses = logits.zip(targets).map(crossEntropy)
    Value.sum(losses) / losses.length.toDouble
