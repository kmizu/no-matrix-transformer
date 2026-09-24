package nomatrix.data

import nomatrix.model.Transformer
import nomatrix.train.Trainer

class PretrainedTest extends munit.FunSuite:

  private val cfg = Corpus.defaultConfig(Corpus.tokenizer.vocabSize)

  test("同梱の学習済みパラメータは既定の Config と同じ形") {
    val expected = Transformer.init(cfg, new scala.util.Random(0)).names.toSet
    for p <- Vector(Pretrained.params, Pretrained.backprop) do
      assertEquals(p.size, Transformer.parameterCount(cfg))
      assertEquals(p.names.toSet, expected)
  }

  test("学習済みパラメータの損失は当てずっぽうよりずっと小さい") {
    val window = Corpus.ids.take(cfg.context + 1)
    for p <- Vector(Pretrained.params, Pretrained.backprop) do
      val loss = Trainer.loss(cfg, p, window)
      assert(loss < math.log(cfg.vocabSize.toDouble) / 2, s"loss=$loss")
  }
