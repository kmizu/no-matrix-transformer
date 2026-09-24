package nomatrix.train

import nomatrix.data.{Corpus, Pretrained}
import nomatrix.model.{FastTransformer, Transformer}
import scala.util.Random

/** 実験: 微分なしの学習がどこまで届くか。
  * `runMain nomatrix.train.Experiment steps pairs sigma lr windows [sgd] [ranks] [out=PATH]`
  */
object Experiment:

  def main(args: Array[String]): Unit =
    val steps = args.lift(0).map(_.toInt).getOrElse(500)
    val pairs = args.lift(1).map(_.toInt).getOrElse(16)
    val sigma = args.lift(2).map(_.toDouble).getOrElse(0.02)
    val lr = args.lift(3).map(_.toDouble).getOrElse(0.01)
    val windows = args.lift(4).map(_.toInt).getOrElse(4)
    val useAdam = !args.contains("sgd")
    val ranks = args.contains("ranks")
    val outPath = args.find(_.startsWith("out=")).map(_.drop(4)).getOrElse("target/evolution-params.txt")

    val tok = Corpus.tokenizer
    val cfg = Corpus.defaultConfig(tok.vocabSize)
    val corpus = Corpus.ids
    val rng = new Random(7)
    val fast = new FastTransformer(cfg)

    // 1. 微分ありの順伝播と、Double だけの順伝播が同じ損失を出すか
    val window = corpus.take(cfg.context + 1)
    val (viaValue, _) = nomatrix.backprop.Trainer.lossAndGradients(cfg, Pretrained.backprop, window)
    val pre = fast.toArray(Pretrained.backprop)
    println(f"equivalence: Value=$viaValue%.6f Plain=${Trainer.loss(cfg, Pretrained.backprop, window)}%.6f Fast=${fast.loss(pre, window)}%.6f")

    // 2. 順伝播 1 回の時間
    val t0 = System.nanoTime()
    var acc = 0.0
    for _ <- 1 to 1000 do acc += fast.loss(pre, window)
    println(f"fast forward: ${(System.nanoTime() - t0) / 1000 / 1e6}%.3f ms (acc=$acc%.1f)")

    // 3. 固定の評価窓（学習には使わない）
    val evalRng = new Random(99)
    val evalWindows = Vector.fill(64)(Trainer.sampleWindow(corpus, cfg.context, evalRng))
    def evalLoss(p: Array[Double]): Double = evalWindows.map(fast.loss(p, _)).sum / evalWindows.length
    println(f"eval loss: random=${evalLoss(fast.toArray(Transformer.init(cfg, new Random(0))))}%.4f  backprop-2000=${evalLoss(pre)}%.4f  guess=${math.log(tok.vocabSize.toDouble)}%.4f")

    // 4. ゆらぎ学習
    val settings = Evolution.Settings(pairs, sigma, lr, useAdam, ranks)
    println(s"evolution: steps=$steps pairs=$pairs sigma=$sigma lr=$lr windows=$windows adam=$useAdam ranks=$ranks")
    val start = System.nanoTime()
    var params = fast.toArray(Transformer.init(cfg, rng))
    var state = Evolution.State.initial(params.length)
    for step <- 1 to steps do
      val batch = Vector.fill(windows)(Trainer.sampleWindow(corpus, cfg.context, rng))
      val lossOf = (p: Array[Double]) => batch.map(fast.loss(p, _)).sum / batch.length
      val (next, nextState, mean) = Evolution.step(params, lossOf, settings, state, rng)
      params = next
      state = nextState
      if step % 250 == 0 || step == 1 then
        println(f"step $step%5d  train ${mean}%.4f  eval ${evalLoss(params)}%.4f  ${(System.nanoTime() - start) / 1e9}%.1fs")
    println(f"final eval ${evalLoss(params)}%.4f in ${(System.nanoTime() - start) / 1e9}%.1fs")
    java.nio.file.Files.writeString(java.nio.file.Path.of(outPath), fast.toParams(params).toText)
