package nomatrix.backprop



import scala.util.Random

class LayersTest extends munit.FunSuite:

  private def rng = new Random(42)

  test("Neuron は重み付き和 + バイアス") {
    val n = Neuron(GVec.fromDoubles(Seq(1.0, 2.0)), Value(0.5))
    assertEqualsDouble(n(GVec.fromDoubles(Seq(3.0, 4.0))).data, 11.5, 1e-9)
  }

  test("Dense.init は in*out + out 個のパラメータを作る") {
    val p = Dense.init("d", in = 3, out = 4, rng)
    assertEquals(p.size, 3 * 4 + 4)
    assert(p.names.forall(_.startsWith("d.")))
  }

  test("Dense は out 個の数を返し、全パラメータに勾配が流れる") {
    val p = Dense.init("d", in = 3, out = 2, rng)
    val pv = p.lift
    val dense = Dense.load(pv, "d", in = 3, out = 2)
    val y = dense(GVec.fromDoubles(Seq(1.0, -1.0, 0.5)))
    assertEquals(y.length, 2)
    val g = pv.gradients(Value.gradients(Value.sum(y)))
    assert(g.keys.forall(_.startsWith("d.")))
    assertEqualsDouble(g("d.n0.b"), 1.0, 1e-9) // バイアスの勾配は必ず 1
    assertEqualsDouble(g("d.n0.w2"), 0.5, 1e-9)
  }

  test("Dense.init の重みは 0 ではなく、大きすぎない") {
    val p = Dense.init("d", in = 16, out = 16, rng)
    val ws = p.names.filter(_.contains(".w")).map(p(_))
    assert(ws.exists(_ != 0.0))
    assert(ws.forall(w => math.abs(w) < 1.0))
  }

  test("Embedding は id から dim 個の数を引く") {
    val p = Embedding.init("e", count = 5, dim = 3, rng)
    assertEquals(p.size, 15)
    val emb = Embedding.load(p.lift, "e", count = 5, dim = 3)
    val v = emb(2)
    assertEquals(v.length, 3)
    assertEqualsDouble(v(1).data, p("e.t2.d1"), 1e-12)
  }

  test("LayerNorm はゲイン 1・バイアス 0 なら GVec.layerNorm と同じ") {
    val p = LayerNorm.init("ln", dim = 4)
    val ln = LayerNorm.load(p.lift, "ln", dim = 4)
    val x = GVec.fromDoubles(Seq(1.0, 2.0, 3.0, 4.0))
    assertEquals(GVec.data(ln(x)), GVec.data(GVec.layerNorm(x)))
  }

  test("AttentionHead は因果的: 未来のトークンを変えても過去の出力は変わらない") {
    val dim = 4
    val p = AttentionHead.init("h", dModel = dim, headDim = 2, rng)
    val head = AttentionHead.load(p.lift, "h", dModel = dim, headDim = 2)
    val xs1 = Vector(Seq(1.0, 0.0, 0.0, 1.0), Seq(0.0, 1.0, 1.0, 0.0), Seq(0.5, 0.5, 0.5, 0.5)).map(GVec.fromDoubles)
    val xs2 = xs1.updated(2, GVec.fromDoubles(Seq(-3.0, 2.0, 9.0, -1.0)))
    val o1 = head(xs1).map(GVec.data)
    val o2 = head(xs2).map(GVec.data)
    assertEquals(o1(0), o2(0))
    assertEquals(o1(1), o2(1))
    assertNotEquals(o1(2), o2(2))
    assertEquals(o1(0).length, 2)
  }

  test("AttentionHead: 1 トークンだけなら出力は value 変換そのもの") {
    val p = AttentionHead.init("h", dModel = 3, headDim = 3, rng)
    val pv = p.lift
    val head = AttentionHead.load(pv, "h", dModel = 3, headDim = 3)
    val x = GVec.fromDoubles(Seq(0.1, 0.2, 0.3))
    val v = Dense.load(pv, "h.value", in = 3, out = 3)(x)
    assertEquals(GVec.data(head(Vector(x))(0)), GVec.data(v))
  }

  test("MultiHead はヘッドを連結して dModel 個の数に戻す") {
    val p = MultiHead.init("mh", dModel = 4, heads = 2, rng)
    val mh = MultiHead.load(p.lift, "mh", dModel = 4, heads = 2)
    val xs = Vector.fill(3)(GVec.fromDoubles(Seq(0.1, -0.2, 0.3, 0.4)))
    val out = mh(xs)
    assertEquals(out.length, 3)
    assert(out.forall(_.length == 4))
  }

  test("MultiHead は dModel がヘッド数で割り切れないと例外") {
    intercept[IllegalArgumentException](MultiHead.init("mh", dModel = 5, heads = 2, rng))
  }

  test("FeedForward は dModel -> hidden -> dModel") {
    val p = FeedForward.init("ff", dModel = 3, hidden = 5, rng)
    assertEquals(p.size, 3 * 5 + 5 + 5 * 3 + 3)
    val ff = FeedForward.load(p.lift, "ff", dModel = 3, hidden = 5)
    assertEquals(ff(GVec.fromDoubles(Seq(1.0, 2.0, 3.0))).length, 3)
  }

  test("Block は形を保ち、残差があるので入力ゼロでもゼロで潰れない") {
    val p = Block.init("b0", dModel = 4, heads = 2, hidden = 8, rng)
    val block = Block.load(p.lift, "b0", dModel = 4, heads = 2, hidden = 8)
    val xs = Vector(Seq(1.0, 2.0, 3.0, 4.0), Seq(4.0, 3.0, 2.0, 1.0)).map(GVec.fromDoubles)
    val out = block(xs)
    assertEquals(out.length, 2)
    assert(out.forall(_.length == 4))
    assert(GVec.data(out(0)).exists(_ != 0.0))
  }
