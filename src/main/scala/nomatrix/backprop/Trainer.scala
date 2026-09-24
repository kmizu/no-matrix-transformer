package nomatrix.backprop


import nomatrix.model.Config
import nomatrix.nn.Params
import scala.util.Random

/** 1 ステップ分の記録。 */
final case class TrainStep(step: Int, loss: Double)

object Trainer:

  /** コーパスから context+1 個の連続した id を切り出す。前 context 個が入力、後ろにずらした context 個が正解。 */
  def sampleWindow(corpus: Vector[Int], context: Int, rng: Random): Vector[Int] =
    require(corpus.length > context, "コーパスが context より短い")
    val start = rng.nextInt(corpus.length - context)
    corpus.slice(start, start + context + 1)

  /** ひとつの窓に対する損失と、名前付きの勾配。 */
  def lossAndGradients(cfg: Config, params: Params, window: Vector[Int]): (Double, Map[String, Double]) =
    val pv = params.lift
    val model = Transformer.load(cfg, pv)
    val inputs = window.dropRight(1)
    val targets = window.drop(1)
    val loss = Loss.sequence(model.logits(inputs), targets)
    (loss.data, pv.gradients(Value.gradients(loss)))

  /** 学習ループ。パラメータは不変なので、各ステップで新しい Params を作って次へ渡す。 */
  def train(
      cfg: Config,
      initial: Params,
      corpus: Vector[Int],
      steps: Int,
      lr: Double,
      rng: Random,
      onStep: TrainStep => Unit = _ => ()
  ): (Params, Vector[TrainStep]) =
    val start = (initial, AdamState.initial, Vector.empty[TrainStep])
    val (params, _, log) = (1 to steps).foldLeft(start) { case ((p, state, log), step) =>
      val window = sampleWindow(corpus, cfg.context, rng)
      val (loss, grads) = lossAndGradients(cfg, p, window)
      val (next, nextState) = Adam.step(p, grads, state, lr)
      val record = TrainStep(step, loss)
      onStep(record)
      (next, nextState, log :+ record)
    }
    (params, log)
