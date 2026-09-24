package nomatrix.backprop

import nomatrix.nn.Params

/** 「微分できる数」に持ち上げたパラメータ。順伝播中はこちらを引く。 */
final case class ParamValues(values: Map[String, Value]):

  def apply(name: String): Value =
    values.getOrElse(name, throw new NoSuchElementException(s"パラメータが無い: $name"))

  /** 逆伝播の結果を「名前 → 勾配」に読み替える。 */
  def gradients(g: Gradients): Map[String, Double] =
    values.map((n, v) => n -> g(v))

object ParamValues:
  /** 順伝播の入口で、数を「微分できる数」に持ち上げる。 */
  def lift(p: Params): ParamValues = ParamValues(p.values.map((n, d) => n -> Value.leaf(d, n)))

extension (p: Params)
  def lift: ParamValues = ParamValues.lift(p)
