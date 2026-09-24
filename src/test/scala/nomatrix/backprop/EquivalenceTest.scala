package nomatrix.backprop

import nomatrix.data.{Corpus, Pretrained}
import nomatrix.nn.Params
import scala.util.Random

/** 本線（Double だけ）と付録（微分できる数）は、同じパラメータで同じ答えを出す。 */
class EquivalenceTest extends munit.FunSuite:

  private val cfg = Corpus.defaultConfig(Corpus.tokenizer.vocabSize)

  test("損失が 1e-9 まで一致する") {
    val rng = new Random(3)
    for p <- Vector(Pretrained.params, Pretrained.backprop); _ <- 1 to 3 do
      val window = nomatrix.train.Trainer.sampleWindow(Corpus.ids, cfg.context, rng)
      val (viaValue, _) = Trainer.lossAndGradients(cfg, p, window)
      assertEqualsDouble(nomatrix.train.Trainer.loss(cfg, p, window), viaValue, 1e-9)
  }

  test("lift した葉は名前を持ち、勾配を名前で取り出せる") {
    val p = Params(Map("w" -> 3.0, "b" -> 1.0, "unused" -> 9.0))
    val pv = p.lift
    assertEquals(pv("w").label, "w")
    val g = pv.gradients(Value.gradients(pv("w") * 2.0 + pv("b")))
    assertEqualsDouble(g("w"), 2.0, 1e-9)
    assertEqualsDouble(g("b"), 1.0, 1e-9)
    assertEqualsDouble(g("unused"), 0.0, 1e-9)
    intercept[NoSuchElementException](pv("nope"))
  }
