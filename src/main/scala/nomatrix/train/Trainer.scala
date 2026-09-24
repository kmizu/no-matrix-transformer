package nomatrix.train

import nomatrix.model.{Config, FastTransformer, Transformer}
import nomatrix.nn.Params
import scala.util.Random

/** 1 ステップ分の記録。`loss` はそのステップで試したゆらぎの平均損失。 */
final case class TrainStep(step: Int, loss: Double)

/** 学習ループ。微分は使わない。「少し動かして試し、良かった方へ寄る」を繰り返す。 */
object Trainer:

  /** コーパスから context+1 個の連続した id を切り出す。前 context 個が入力、後ろにずらした context 個が正解。 */
  def sampleWindow(corpus: Vector[Int], context: Int, rng: Random): Vector[Int] =
    require(corpus.length > context, "コーパスが context より短い")
    val start = rng.nextInt(corpus.length - context)
    corpus.slice(start, start + context + 1)

  /** ひとつの窓に対する損失（読みやすい版のモデルで計算）。 */
  def loss(cfg: Config, params: Params, window: Vector[Int]): Double =
    val model = Transformer.load(cfg, params)
    Loss.sequence(model.logits(window.dropRight(1)), window.drop(1))

  def train(
      cfg: Config,
      initial: Params,
      corpus: Vector[Int],
      steps: Int,
      settings: Evolution.Settings = Evolution.Settings(),
      windows: Int = 8,
      rng: Random = new Random(0),
      onStep: TrainStep => Unit = _ => ()
  ): (Params, Vector[TrainStep]) =
    val fast = new FastTransformer(cfg)
    val start = (fast.toArray(initial), Evolution.State.initial(fast.size), Vector.empty[TrainStep])
    val (params, _, log) = (1 to steps).foldLeft(start) { case ((p, state, log), step) =>
      // 同じ窓を全部のゆらぎに使う。比べる相手が同じでないと「どちらが良いか」が分からない
      val batch = Vector.fill(windows)(sampleWindow(corpus, cfg.context, rng))
      val lossOf = (candidate: Array[Double]) => batch.map(fast.loss(candidate, _)).sum / batch.length
      val (next, nextState, mean) = Evolution.step(p, lossOf, settings, state, rng)
      val record = TrainStep(step, mean)
      onStep(record)
      (next, nextState, log :+ record)
    }
    (fast.toParams(params), log)
