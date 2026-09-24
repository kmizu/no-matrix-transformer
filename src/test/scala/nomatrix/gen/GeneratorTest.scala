package nomatrix.gen

import nomatrix.model.{Config, Transformer}
import nomatrix.vec.Vec
import scala.util.Random

class GeneratorTest extends munit.FunSuite:

  private val cfg = Config(vocabSize = 5, dModel = 4, heads = 2, layers = 1, context = 4, hidden = 8)

  test("指定した数だけトークンを足し、全部語彙の範囲内") {
    val p = Transformer.init(cfg, new Random(1))
    val out = Generator.generate(cfg, p, prompt = Vector(0), count = 6, temperature = 1.0, rng = new Random(2))
    assertEquals(out.length, 7)
    assert(out.forall(id => id >= 0 && id < cfg.vocabSize))
    assertEquals(out.head, 0)
  }

  test("プロンプトが context より長くても末尾だけを見て続けられる") {
    val p = Transformer.init(cfg, new Random(1))
    val out = Generator.generate(cfg, p, prompt = Vector(0, 1, 2, 3, 4, 0), count = 2, temperature = 1.0, rng = new Random(2))
    assertEquals(out.length, 8)
  }

  test("温度 0 なら最大のロジットを選ぶ") {
    val logits = Vec.fromDoubles(Seq(0.1, 5.0, -2.0))
    assertEquals(Generator.sample(logits, temperature = 0.0, new Random(0)), 1)
  }

  test("温度 1 のサンプリングは分布に従う") {
    val logits = Vec.fromDoubles(Seq(0.0, 10.0))
    val picks = (1 to 200).map(_ => Generator.sample(logits, temperature = 1.0, new Random(7)))
    assert(picks.count(_ == 1) > 190)
  }
