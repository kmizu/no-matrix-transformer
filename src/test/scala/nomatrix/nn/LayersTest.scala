package nomatrix.nn

import scala.util.Random

class LayersTest extends munit.FunSuite:

  private def rng = new Random(42)

  test("Neuron は重み付き和 + バイアス") {
    val n = Neuron(Vector(1.0, 2.0), 0.5)
    assertEqualsDouble(n(Vector(3.0, 4.0)), 11.5, 1e-9)
  }

  test("Dense.init は in*out + out 個のパラメータを作り、重みは 0 でなく大きすぎない") {
    val p = Dense.init("d", in = 16, out = 16, rng)
    assertEquals(p.size, 16 * 16 + 16)
    assert(p.names.forall(_.startsWith("d.")))
    val ws = p.names.filter(_.contains(".w")).map(p(_))
    assert(ws.exists(_ != 0.0))
    assert(ws.forall(w => math.abs(w) < 1.0))
  }

  test("Dense は out 個の数を返し、重みを名前で動かすと出力が動く") {
    val p = Dense.init("d", in = 3, out = 2, rng)
    val x = Vector(1.0, -1.0, 0.5)
    val before = Dense.load(p, "d", 3, 2)(x)
    assertEquals(before.length, 2)
    val after = Dense.load(p.updated("d.n0.w2", p("d.n0.w2") + 1.0), "d", 3, 2)(x)
    assertEqualsDouble(after(0) - before(0), 0.5, 1e-9) // 入力の 2 番目 (0.5) × 増やした重み (1.0)
    assertEqualsDouble(after(1), before(1), 1e-12)
  }

  test("Embedding は id から dim 個の数を引く") {
    val p = Embedding.init("e", count = 5, dim = 3, rng)
    assertEquals(p.size, 15)
    val emb = Embedding.load(p, "e", count = 5, dim = 3)
    assertEquals(emb(2).length, 3)
    assertEqualsDouble(emb(2)(1), p("e.t2.d1"), 1e-12)
  }

  test("LayerNorm はゲイン 1・バイアス 0 なら Vec.layerNorm と同じ") {
    val ln = LayerNorm.load(LayerNorm.init("ln", dim = 4), "ln", dim = 4)
    val x = Vector(1.0, 2.0, 3.0, 4.0)
    assertEquals(ln(x), nomatrix.vec.Vec.layerNorm(x))
  }

  test("AttentionHead は因果的: 未来のトークンを変えても過去の出力は変わらない") {
    val head = AttentionHead.load(AttentionHead.init("h", dModel = 4, headDim = 2, rng), "h", 4, 2)
    val xs1 = Vector(Vector(1.0, 0.0, 0.0, 1.0), Vector(0.0, 1.0, 1.0, 0.0), Vector(0.5, 0.5, 0.5, 0.5))
    val xs2 = xs1.updated(2, Vector(-3.0, 2.0, 9.0, -1.0))
    val (o1, o2) = (head(xs1), head(xs2))
    assertEquals(o1(0), o2(0))
    assertEquals(o1(1), o2(1))
    assertNotEquals(o1(2), o2(2))
    assertEquals(o1(0).length, 2)
  }

  test("AttentionHead: 1 トークンだけなら出力は value 変換そのもので、注目の割合は 1") {
    val p = AttentionHead.init("h", dModel = 3, headDim = 3, rng)
    val head = AttentionHead.load(p, "h", 3, 3)
    val x = Vector(0.1, 0.2, 0.3)
    assertEquals(head(Vector(x))(0), Dense.load(p, "h.value", 3, 3)(x))
    assertEquals(head.weights(Vector(x), 0), Vector(1.0))
  }

  test("MultiHead はヘッドを連結して dModel 個の数に戻し、割り切れない設定は例外") {
    val mh = MultiHead.load(MultiHead.init("mh", dModel = 4, heads = 2, rng), "mh", 4, 2)
    val out = mh(Vector.fill(3)(Vector(0.1, -0.2, 0.3, 0.4)))
    assertEquals(out.length, 3)
    assert(out.forall(_.length == 4))
    intercept[IllegalArgumentException](MultiHead.init("mh", dModel = 5, heads = 2, rng))
  }

  test("FeedForward は dModel -> hidden -> dModel") {
    val p = FeedForward.init("ff", dModel = 3, hidden = 5, rng)
    assertEquals(p.size, 3 * 5 + 5 + 5 * 3 + 3)
    assertEquals(FeedForward.load(p, "ff", 3, 5)(Vector(1.0, 2.0, 3.0)).length, 3)
  }

  test("Block は形を保ち、残差があるのでゼロで潰れない") {
    val block = Block.load(Block.init("b0", dModel = 4, heads = 2, hidden = 8, rng), "b0", 4, 2, 8)
    val out = block(Vector(Vector(1.0, 2.0, 3.0, 4.0), Vector(4.0, 3.0, 2.0, 1.0)))
    assertEquals(out.length, 2)
    assert(out.forall(_.length == 4))
    assert(out(0).exists(_ != 0.0))
  }
