package nomatrix.train

import nomatrix.autograd.Value
import nomatrix.nn.Params
import nomatrix.vec.Vec
import nomatrix.model.{Config, Transformer}
import scala.util.Random

class LossTest extends munit.FunSuite:

  test("交差エントロピーは -log(softmax の正解確率)") {
    val logits = Vec.fromDoubles(Seq(1.0, 2.0, 3.0))
    val probs = Vec.data(Vec.softmax(logits))
    assertEqualsDouble(Loss.crossEntropy(logits, 2).data, -math.log(probs(2)), 1e-9)
  }

  test("大きなロジットでも NaN にならない") {
    val logits = Vec.fromDoubles(Seq(1000.0, -1000.0))
    val l = Loss.crossEntropy(logits, 1).data
    assert(!l.isNaN && l > 1000.0)
  }

  test("交差エントロピーの勾配は softmax - onehot") {
    val logits = Vec.fromDoubles(Seq(0.5, -1.0, 2.0))
    val probs = Vec.data(Vec.softmax(logits))
    val g = Value.gradients(Loss.crossEntropy(logits, 0))
    assertEqualsDouble(g(logits(0)), probs(0) - 1.0, 1e-9)
    assertEqualsDouble(g(logits(1)), probs(1), 1e-9)
  }

  test("列の損失は各位置の平均") {
    val a = Vec.fromDoubles(Seq(1.0, 0.0))
    val b = Vec.fromDoubles(Seq(0.0, 1.0))
    val expected = (Loss.crossEntropy(a, 0).data + Loss.crossEntropy(b, 0).data) / 2
    assertEqualsDouble(Loss.sequence(Vector(a, b), Vector(0, 0)).data, expected, 1e-9)
  }

class AdamTest extends munit.FunSuite:

  test("Adam は二次関数の最小点へ向かう") {
    val start = Params(Map("x" -> 5.0, "y" -> -3.0))
    val (end, _) = (1 to 300).foldLeft((start, AdamState.initial)) { case ((p, s), _) =>
      val pv = p.lift
      val loss = pv("x") * pv("x") + (pv("y") - 1.0) * (pv("y") - 1.0)
      Adam.step(p, pv.gradients(Value.gradients(loss)), s, lr = 0.1)
    }
    assertEqualsDouble(end("x"), 0.0, 0.05)
    assertEqualsDouble(end("y"), 1.0, 0.05)
  }

  test("1 ステップ目の更新量は符号 × lr にほぼ等しい") {
    val p = Params(Map("x" -> 1.0))
    val (next, state) = Adam.step(p, Map("x" -> 123.0), AdamState.initial, lr = 0.01)
    assertEqualsDouble(next("x"), 0.99, 1e-6)
    assertEquals(state.step, 1)
  }

class TrainerTest extends munit.FunSuite:

  test("小さなコーパスを数十ステップ学習すると損失が下がる") {
    val cfg = Config(vocabSize = 4, dModel = 4, heads = 2, layers = 1, context = 4, hidden = 8)
    val corpus = Vector.tabulate(64)(i => i % 4) // 0 1 2 3 0 1 2 3 ...
    val rng = new Random(0)
    val init = Transformer.init(cfg, rng)
    val (_, log) = Trainer.train(cfg, init, corpus, steps = 40, lr = 0.05, rng = rng)
    assertEquals(log.length, 40)
    val first = log.take(5).map(_.loss).sum / 5
    val last = log.takeRight(5).map(_.loss).sum / 5
    assert(last < first, s"loss did not decrease: $first -> $last")
  }

  test("sampleWindow は context+1 個の連続した id を返す") {
    val corpus = Vector.range(0, 20)
    val w = Trainer.sampleWindow(corpus, context = 5, new Random(1))
    assertEquals(w.length, 6)
    assertEquals(w, Vector.range(w.head, w.head + 6))
  }
