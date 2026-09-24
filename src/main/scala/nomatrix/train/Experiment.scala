package nomatrix.train

import nomatrix.data.{Corpus, Pretrained}
import nomatrix.model.{FastTransformer, Transformer}
import nomatrix.nn.Params
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
    val lineSearch = args.contains("ls")
    val outPath = args.find(_.startsWith("out=")).map(_.drop(4)).getOrElse("target/evolution-params.txt")

    val tok = Corpus.tokenizer
    val cfg = Corpus.defaultConfig(tok.vocabSize)
    val corpus = Corpus.ids
    val rng = new Random(7)
    val fast = new FastTransformer(cfg)
    // groups: 名前の先頭で 4 組（埋め込み / block0 / block1 / 仕上げの norm と head）
    val groups: Vector[Vector[Int]] =
      if args.contains("groups") then
        def groupOf(name: String): Int = name.split('.').head match
          case "tok" | "pos" => 0
          case "block0"      => 1
          case "block1"      => 2
          case _             => 3
        fast.names.zipWithIndex.groupBy((name, _) => groupOf(name)).toVector.sortBy(_._1).map(_._2.map(_._2))
      else Vector.empty

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
    val settings = Evolution.Settings(pairs, sigma, lr, useAdam, ranks, groups, lineSearch)
    // zero: 残差の枝（attn.output と ff.down の重み）を 0 から始める。各ブロックは最初「素通し」
    val zeroResidual = args.contains("zero")
    // stages: 前半は 埋め込み+block0+仕上げ だけ、後半は block1+仕上げ だけを育てる（育てながら積む）
    val stages = args.contains("stages")
    def indicesOf(pred: String => Boolean): Vector[Int] = fast.names.zipWithIndex.collect { case (n, i) if pred(n) => i }
    val stage1 = indicesOf(n => !n.startsWith("block1"))
    val stage2 = indicesOf(n => n.startsWith("block1") || n.startsWith("norm") || n.startsWith("head"))
    println(s"evolution: steps=$steps pairs=$pairs sigma=$sigma lr=$lr windows=$windows adam=$useAdam ranks=$ranks groups=${groups.map(_.size)} ls=$lineSearch zero=$zeroResidual stages=$stages")
    val start = System.nanoTime()
    val initial = Transformer.init(cfg, rng)
    var params = fast.toArray(
      if zeroResidual then Params(initial.values.map((n, v) => n -> (if (n.contains(".attn.output.") || n.contains(".ff.down.")) && n.contains(".w") then 0.0 else v)))
      else initial)
    var state = Evolution.State.initial(params.length)
    for step <- 1 to steps do
      val batch = Vector.fill(windows)(Trainer.sampleWindow(corpus, cfg.context, rng))
      val lossOf = (p: Array[Double]) => batch.map(fast.loss(p, _)).sum / batch.length
      val current =
        if !stages then settings
        else if step <= steps / 2 then settings.copy(groups = Vector(stage1))
        else settings.copy(groups = Vector(stage2))
      val (next, nextState, mean) = Evolution.step(params, lossOf, current, state, rng)
      params = next
      state = nextState
      if step % 250 == 0 || step == 1 then
        println(f"step $step%5d  train ${mean}%.4f  eval ${evalLoss(params)}%.4f  ${(System.nanoTime() - start) / 1e9}%.1fs")
    println(f"final eval ${evalLoss(params)}%.4f in ${(System.nanoTime() - start) / 1e9}%.1fs")
    java.nio.file.Files.writeString(java.nio.file.Path.of(outPath), fast.toParams(params).toText)
