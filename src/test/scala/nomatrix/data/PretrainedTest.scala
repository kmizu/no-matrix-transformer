package nomatrix.data

import nomatrix.model.Transformer
import nomatrix.train.Trainer

class PretrainedTest extends munit.FunSuite:

  test("同梱の学習済みパラメータは既定の Config と同じ形") {
    val cfg = Corpus.defaultConfig(Corpus.tokenizer.vocabSize)
    assertEquals(Pretrained.params.size, Transformer.parameterCount(cfg))
    assertEquals(Pretrained.params.names.toSet, Transformer.init(cfg, new scala.util.Random(0)).names.toSet)
  }

  test("学習済みパラメータの損失は当てずっぽうよりずっと小さい") {
    val cfg = Corpus.defaultConfig(Corpus.tokenizer.vocabSize)
    val window = Corpus.ids.take(cfg.context + 1)
    val (loss, _) = Trainer.lossAndGradients(cfg, Pretrained.params, window)
    assert(loss < math.log(cfg.vocabSize.toDouble) / 2, s"loss=$loss")
  }
