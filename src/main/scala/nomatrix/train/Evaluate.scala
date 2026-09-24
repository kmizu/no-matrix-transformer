package nomatrix.train

import nomatrix.data.Corpus
import nomatrix.gen.Generator
import nomatrix.model.FastTransformer
import nomatrix.nn.Params
import java.nio.file.{Files, Path}
import scala.util.Random

/** パラメータファイルを、学習に使っていない固定の 64 窓で評価する。`runMain nomatrix.train.Evaluate FILE...` */
object Evaluate:

  def main(args: Array[String]): Unit =
    val tok = Corpus.tokenizer
    val cfg = Corpus.defaultConfig(tok.vocabSize)
    val fast = new FastTransformer(cfg)
    val evalRng = new Random(99)
    val evalWindows = Vector.fill(64)(Trainer.sampleWindow(Corpus.ids, cfg.context, evalRng))
    for path <- args do
      val params = Params.parse(Files.readString(Path.of(path)))
      val arr = fast.toArray(params)
      val loss = evalWindows.map(fast.loss(arr, _)).sum / evalWindows.length
      val sample = tok.decode(Generator.generate(cfg, params, tok.encode("あさ"), 60, 0.5, new Random(1)))
      println(f"$path  eval=$loss%.4f  sample=${sample.replace("\n", "⏎")}")
