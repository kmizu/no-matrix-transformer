package nomatrix.model

import nomatrix.data.Pretrained
import nomatrix.nn.Params
import scala.util.Random

class TransformerTest extends munit.FunSuite:

  private val cfg = Config(vocabSize = 7, dModel = 4, heads = 2, layers = 2, context = 5, hidden = 8)

  test("パラメータ数が手計算と一致する") {
    val dense = (in: Int, out: Int) => in * out + out
    val block =
      2 * (2 * cfg.dModel) +
        cfg.heads * 3 * dense(cfg.dModel, cfg.dModel / cfg.heads) + dense(cfg.dModel, cfg.dModel) +
        dense(cfg.dModel, cfg.hidden) + dense(cfg.hidden, cfg.dModel)
    val expected =
      cfg.vocabSize * cfg.dModel + cfg.context * cfg.dModel + cfg.layers * block + 2 * cfg.dModel + dense(cfg.dModel, cfg.vocabSize)
    assertEquals(Transformer.init(cfg, new Random(1)).size, expected)
    assertEquals(Transformer.parameterCount(cfg), expected)
  }

  test("logits は各位置ごとに vocabSize 個の数") {
    val model = Transformer.load(cfg, Transformer.init(cfg, new Random(1)))
    val out = model.logits(Vector(1, 2, 3))
    assertEquals(out.length, 3)
    assert(out.forall(_.length == cfg.vocabSize))
  }

  test("context を超える長さは例外、語彙外の id も例外") {
    val model = Transformer.load(cfg, Transformer.init(cfg, new Random(1)))
    intercept[IllegalArgumentException](model.logits(Vector.fill(cfg.context + 1)(0)))
    intercept[IllegalArgumentException](model.logits(Vector(cfg.vocabSize)))
  }

  test("FastTransformer は読みやすい版と同じロジットを出し、配列と辞書を往復できる") {
    val real = nomatrix.data.Corpus.defaultConfig(nomatrix.data.Corpus.tokenizer.vocabSize)
    val fast = new FastTransformer(real)
    val p: Params = Pretrained.backprop
    assertEquals(fast.toParams(fast.toArray(p)), p)
    assertEquals(fast.size, p.size)
    val ids = Vector(3, 1, 4, 1, 5)
    val slow = Transformer.load(real, p).logits(ids)
    val quick = fast.logits(fast.toArray(p), ids)
    for (a, b) <- slow.zip(quick); (x, y) <- a.zip(b) do assertEqualsDouble(x, y, 1e-9)
  }
