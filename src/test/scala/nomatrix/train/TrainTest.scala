package nomatrix.train

import nomatrix.model.{Config, Transformer}
import nomatrix.vec.Vec
import scala.util.Random

class LossTest extends munit.FunSuite:

  test("交差エントロピーは -log(softmax の正解確率)") {
    val logits = Vector(1.0, 2.0, 3.0)
    val probs = Vec.softmax(logits)
    assertEqualsDouble(Loss.crossEntropy(logits, 2), -math.log(probs(2)), 1e-9)
  }

  test("大きなロジットでも NaN にならない") {
    val l = Loss.crossEntropy(Vector(1000.0, -1000.0), 1)
    assert(!l.isNaN && l > 1000.0)
  }

  test("列の損失は各位置の平均") {
    val a = Vector(1.0, 0.0)
    val b = Vector(0.0, 1.0)
    val expected = (Loss.crossEntropy(a, 0) + Loss.crossEntropy(b, 0)) / 2
    assertEqualsDouble(Loss.sequence(Vector(a, b), Vector(0, 0)), expected, 1e-9)
    intercept[IllegalArgumentException](Loss.sequence(Vector(a), Vector(0, 0)))
  }

class EvolutionTest extends munit.FunSuite:

  private def minimize(lossOf: Array[Double] => Double, start: Array[Double], settings: Evolution.Settings, steps: Int): Array[Double] =
    val rng = new Random(0)
    var params = start
    var state = Evolution.State.initial(start.length)
    for _ <- 1 to steps do
      val (next, nextState, _) = Evolution.step(params, lossOf, settings, state, rng)
      params = next
      state = nextState
    params

  test("ゆらぎ学習は二次関数の最小点へ向かう（微分なし）") {
    val lossOf = (p: Array[Double]) => (p(0) - 1.0) * (p(0) - 1.0) + (p(1) + 2.0) * (p(1) + 2.0)
    val end = minimize(lossOf, Array(5.0, 5.0), Evolution.Settings(pairs = 8, sigma = 0.1, lr = 0.02), 800)
    assertEqualsDouble(end(0), 1.0, 0.2)
    assertEqualsDouble(end(1), -2.0, 0.2)
  }

  test("順位重み版と Adam なし版も同じ最小点へ向かう") {
    val lossOf = (p: Array[Double]) => (p(0) - 1.0) * (p(0) - 1.0)
    val ranked = minimize(lossOf, Array(-3.0), Evolution.Settings(pairs = 8, sigma = 0.1, lr = 0.02, ranks = true), 800)
    assertEqualsDouble(ranked(0), 1.0, 0.2)
    val plain = minimize(lossOf, Array(-3.0), Evolution.Settings(pairs = 8, sigma = 0.1, lr = 0.05, useAdam = false), 300)
    assertEqualsDouble(plain(0), 1.0, 0.2)
  }

  test("groups を指定すると、その組のパラメータしか動かない") {
    val lossOf = (p: Array[Double]) => p.map(x => x * x).sum
    val settings = Evolution.Settings(pairs = 4, sigma = 0.1, lr = 0.05, groups = Vector(Vector(0), Vector(1)))
    val start = Array(3.0, 3.0, 3.0)
    val (after1, state1, _) = Evolution.step(start, lossOf, settings, Evolution.State.initial(3), new Random(0))
    assert(after1(0) != 3.0 && after1(1) == 3.0 && after1(2) == 3.0)
    val (after2, _, _) = Evolution.step(after1, lossOf, settings, state1, new Random(0))
    assert(after2(0) == after1(0) && after2(1) != 3.0 && after2(2) == 3.0)
  }

  test("lineSearch は歩幅を半分・そのまま・倍から選び、損失が下がらない歩幅は選ばない") {
    val lossOf = (p: Array[Double]) => (p(0) - 1.0) * (p(0) - 1.0)
    val plain = minimize(lossOf, Array(-3.0), Evolution.Settings(pairs = 8, sigma = 0.1, lr = 0.02), 100)
    val searched = minimize(lossOf, Array(-3.0), Evolution.Settings(pairs = 8, sigma = 0.1, lr = 0.02, lineSearch = true), 100)
    assert(math.abs(searched(0) - 1.0) <= math.abs(plain(0) - 1.0) + 0.1, s"plain=${plain(0)} searched=${searched(0)}")
  }

  test("並列評価は逐次評価と同じ結果になる（乱数は逐次に引くので決定的）") {
    val lossOf = (p: Array[Double]) => p.zipWithIndex.map((x, i) => (x - i) * (x - i)).sum
    val start = Array(5.0, 5.0, 5.0, 5.0)
    val serial = Evolution.step(start, lossOf, Evolution.Settings(pairs = 8, sigma = 0.1, lr = 0.05), Evolution.State.initial(4), new Random(3))._1
    val parallel = Evolution.step(start, lossOf, Evolution.Settings(pairs = 8, sigma = 0.1, lr = 0.05, parallel = true), Evolution.State.initial(4), new Random(3))._1
    assertEquals(parallel.toVector, serial.toVector)
  }

  test("直交化・±1・曲率のゆらぎでも最小点へ向かう") {
    val lossOf = (p: Array[Double]) => (p(0) - 1.0) * (p(0) - 1.0) + (p(1) + 2.0) * (p(1) + 2.0)
    for settings <- Vector(
        Evolution.Settings(pairs = 8, sigma = 0.1, lr = 0.02, orthogonal = true),
        Evolution.Settings(pairs = 8, sigma = 0.1, lr = 0.02, rademacher = true),
        Evolution.Settings(pairs = 8, sigma = 0.1, lr = 0.02, newton = true))
    do
      val end = minimize(lossOf, Array(5.0, 5.0), settings, 1200)
      assertEqualsDouble(end(0), 1.0, 0.3, settings.toString)
      assertEqualsDouble(end(1), -2.0, 0.3, settings.toString)
  }

  test("ステップは試したゆらぎの平均損失を返す") {
    val (_, _, mean) = Evolution.step(Array(0.0), p => p(0) * p(0), Evolution.Settings(pairs = 4, sigma = 1.0), Evolution.State.initial(1), new Random(1))
    assert(mean > 0.0)
  }

class TrainerTest extends munit.FunSuite:

  test("sampleWindow は context+1 個の連続した id を返す") {
    val w = Trainer.sampleWindow(Vector.range(0, 20), context = 5, new Random(1))
    assertEquals(w.length, 6)
    assertEquals(w, Vector.range(w.head, w.head + 6))
    intercept[IllegalArgumentException](Trainer.sampleWindow(Vector(1, 2), context = 5, new Random(1)))
  }

  test("小さなコーパスを数百ステップ学習すると損失が下がる（微分なし）") {
    val cfg = Config(vocabSize = 4, dModel = 4, heads = 2, layers = 1, context = 4, hidden = 8)
    val corpus = Vector.tabulate(64)(i => i % 4)
    val rng = new Random(0)
    val init = Transformer.init(cfg, rng)
    val (trained, log) = Trainer.train(cfg, init, corpus, steps = 300, Evolution.Settings(pairs = 16, sigma = 0.05, lr = 0.02), windows = 4, rng)
    assertEquals(log.length, 300)
    val window = Vector(0, 1, 2, 3, 0)
    val before = Trainer.loss(cfg, init, window)
    val after = Trainer.loss(cfg, trained, window)
    assert(after < before / 2, s"loss did not fall enough: $before -> $after")
  }
