package nomatrix.backprop

import nomatrix.nn.Params



import scala.util.Random

/** ニューロン = 入力の重み付き和にバイアスを足すだけの関数。 */
final case class Neuron(weights: GVec, bias: Value):
  def apply(x: GVec): Value = GVec.dot(weights, x) + bias

/** Dense = ニューロンの集まり。入力ベクトルを各ニューロンに見せて、答えを並べる。
  *
  * 世間では「行列とベクトルの積」と呼ばれる操作だが、ここでは「ニューロンが並んでいる」だけ。
  */
final case class Dense(neurons: Vector[Neuron]):
  def apply(x: GVec): GVec = neurons.map(_(x))

object Dense:

  private def weightName(prefix: String, j: Int, i: Int) = s"$prefix.n$j.w$i"
  private def biasName(prefix: String, j: Int) = s"$prefix.n$j.b"

  /** 重みは入力数に応じた小さな乱数、バイアスは 0 から始める。 */
  def init(prefix: String, in: Int, out: Int, rng: Random): Params =
    val bound = 1.0 / math.sqrt(in.toDouble)
    val weights =
      for j <- 0 until out; i <- 0 until in
      yield weightName(prefix, j, i) -> (rng.nextDouble() * 2 - 1) * bound
    val biases = (0 until out).map(j => biasName(prefix, j) -> 0.0)
    Params((weights ++ biases).toMap)

  def load(p: ParamValues, prefix: String, in: Int, out: Int): Dense =
    Dense(
      (0 until out).toVector.map { j =>
        Neuron((0 until in).toVector.map(i => p(weightName(prefix, j, i))), p(biasName(prefix, j)))
      }
    )
