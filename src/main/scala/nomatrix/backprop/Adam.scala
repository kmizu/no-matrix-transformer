package nomatrix.backprop

import nomatrix.nn.Params

/** Adam の「これまでの勾配の記憶」。名前ごとに、勾配の平均（m）と二乗平均（v）を持つ。 */
final case class AdamState(step: Int, m: Map[String, Double], v: Map[String, Double])

object AdamState:
  val initial: AdamState = AdamState(0, Map.empty, Map.empty)

/** Adam 最適化。パラメータも状態も不変で、更新のたびに新しいものを返す。 */
object Adam:

  def step(
      params: Params,
      grads: Map[String, Double],
      state: AdamState,
      lr: Double,
      beta1: Double = 0.9,
      beta2: Double = 0.999,
      eps: Double = 1e-8
  ): (Params, AdamState) =
    val t = state.step + 1
    val updated = params.values.map { (name, value) =>
      val g = grads.getOrElse(name, 0.0)
      val m = beta1 * state.m.getOrElse(name, 0.0) + (1 - beta1) * g
      val v = beta2 * state.v.getOrElse(name, 0.0) + (1 - beta2) * g * g
      val mHat = m / (1 - math.pow(beta1, t))
      val vHat = v / (1 - math.pow(beta2, t))
      name -> (value - lr * mHat / (math.sqrt(vHat) + eps), m, v)
    }
    (
      Params(updated.map((n, r) => n -> r._1)),
      AdamState(t, updated.map((n, r) => n -> r._2), updated.map((n, r) => n -> r._3))
    )
