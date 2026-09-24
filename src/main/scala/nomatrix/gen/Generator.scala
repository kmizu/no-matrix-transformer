package nomatrix.gen

import nomatrix.model.{Config, Transformer}
import nomatrix.nn.Params
import nomatrix.vec.Vec
import scala.util.Random

object Generator:

  /** ロジットから次のトークンをひとつ選ぶ。温度 0 なら最大、そうでなければ softmax の割合で抽選。 */
  def sample(logits: Vec, temperature: Double, rng: Random): Int =
    require(!temperature.isNaN, "温度が NaN")
    val scores = Vec.data(logits)
    val argmax = scores.indices.maxBy(scores)
    val scaled = scores.map(_ / temperature)
    if temperature <= 0.0 || scaled.exists(s => s.isNaN || s.isInfinite) then argmax
    else
      val probs = Vec.data(Vec.softmax(Vec.fromDoubles(scaled)))
      val r = rng.nextDouble()
      val cumulative = probs.scanLeft(0.0)(_ + _).tail
      cumulative.indexWhere(_ > r) match
        case -1 => argmax
        case i  => i

  /** プロンプトの続きを count トークンぶん生成する。context を超えたら末尾だけを見る。 */
  def generate(cfg: Config, params: Params, prompt: Vector[Int], count: Int, temperature: Double, rng: Random): Vector[Int] =
    require(prompt.nonEmpty, "プロンプトが空")
    val model = Transformer.load(cfg, params.lift)
    (1 to count).foldLeft(prompt) { (ids, _) =>
      val window = ids.takeRight(cfg.context)
      val next = sample(model.logits(window).last, temperature, rng)
      ids :+ next
    }
