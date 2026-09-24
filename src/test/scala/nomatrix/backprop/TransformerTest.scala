package nomatrix.backprop

import nomatrix.model.Config


import scala.util.Random

class TransformerTest extends munit.FunSuite:

  private val cfg = Config(vocabSize = 7, dModel = 4, heads = 2, layers = 2, context = 5, hidden = 8)

  test("パラメータ数が手計算と一致する") {
    val p = Transformer.init(cfg, new Random(1))
    val dense = (in: Int, out: Int) => in * out + out
    val block =
      2 * (2 * cfg.dModel) + // norm1, norm2
        cfg.heads * 3 * dense(cfg.dModel, cfg.dModel / cfg.heads) + dense(cfg.dModel, cfg.dModel) + // attention
        dense(cfg.dModel, cfg.hidden) + dense(cfg.hidden, cfg.dModel) // ff
    val expected =
      cfg.vocabSize * cfg.dModel + cfg.context * cfg.dModel + cfg.layers * block + 2 * cfg.dModel + dense(cfg.dModel, cfg.vocabSize)
    assertEquals(p.size, expected)
    assertEquals(Transformer.parameterCount(cfg), expected)
  }

  test("logits は各位置ごとに vocabSize 個の数") {
    val p = Transformer.init(cfg, new Random(1))
    val model = Transformer.load(cfg, p.lift)
    val out = model.logits(Vector(1, 2, 3))
    assertEquals(out.length, 3)
    assert(out.forall(_.length == cfg.vocabSize))
  }

  test("context を超える長さは例外、語彙外の id も例外") {
    val model = Transformer.load(cfg, Transformer.init(cfg, new Random(1)).lift)
    intercept[IllegalArgumentException](model.logits(Vector.fill(cfg.context + 1)(0)))
    intercept[IllegalArgumentException](model.logits(Vector(cfg.vocabSize)))
  }

  test("全パラメータに勾配が流れる（出力ヘッドと埋め込みの一部を確認）") {
    val p = Transformer.init(cfg, new Random(3))
    val pv = p.lift
    val model = Transformer.load(cfg, pv)
    val out = model.logits(Vector(0, 1))
    val loss = Value.sum(out.flatten)
    val g = pv.gradients(Value.gradients(loss))
    assert(g("head.n0.b") != 0.0)
    assert(g("tok.t1.d0") != 0.0)
    assert(g("pos.t0.d0") != 0.0)
    assert(g("block1.ff.up.n0.w0") != 0.0)
  }
