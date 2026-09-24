package nomatrix.vec

import nomatrix.autograd.Value

class VecTest extends munit.FunSuite:
  private val eps = 1e-9

  test("fromDoubles / data で往復できる") {
    val v = Vec.fromDoubles(Seq(1.0, 2.0, 3.0))
    assertEquals(Vec.data(v), Vector(1.0, 2.0, 3.0))
  }

  test("add / sub / scale は要素ごと") {
    val a = Vec.fromDoubles(Seq(1.0, 2.0))
    val b = Vec.fromDoubles(Seq(10.0, 20.0))
    assertEquals(Vec.data(Vec.add(a, b)), Vector(11.0, 22.0))
    assertEquals(Vec.data(Vec.sub(b, a)), Vector(9.0, 18.0))
    assertEquals(Vec.data(Vec.scale(a, 3.0)), Vector(3.0, 6.0))
  }

  test("mul は要素ごとの掛け算") {
    val a = Vec.fromDoubles(Seq(2.0, 3.0))
    val b = Vec.fromDoubles(Seq(4.0, 5.0))
    assertEquals(Vec.data(Vec.mul(a, b)), Vector(8.0, 15.0))
  }

  test("長さが違うベクトルの add は例外") {
    val a = Vec.fromDoubles(Seq(1.0))
    val b = Vec.fromDoubles(Seq(1.0, 2.0))
    intercept[IllegalArgumentException](Vec.add(a, b))
  }

  test("dot は積の和") {
    val a = Vec.fromDoubles(Seq(1.0, 2.0, 3.0))
    val b = Vec.fromDoubles(Seq(4.0, 5.0, 6.0))
    assertEqualsDouble(Vec.dot(a, b).data, 32.0, eps)
  }

  test("softmax は合計 1 で、大きい値ほど大きい割合") {
    val v = Vec.fromDoubles(Seq(1.0, 2.0, 3.0))
    val s = Vec.data(Vec.softmax(v))
    assertEqualsDouble(s.sum, 1.0, eps)
    assert(s(2) > s(1) && s(1) > s(0))
  }

  test("softmax は大きな値でもオーバーフローしない") {
    val v = Vec.fromDoubles(Seq(1000.0, 1000.0))
    val s = Vec.data(Vec.softmax(v))
    assertEqualsDouble(s(0), 0.5, eps)
  }

  test("softmax の勾配が数値微分と一致する") {
    val xs = Vector(0.2, -0.5, 1.1)
    def f(ds: Vector[Double]): Double =
      val s = Vec.data(Vec.softmax(Vec.fromDoubles(ds)))
      s(0) * 1.0 + s(1) * 2.0 + s(2) * 3.0
    val v = Vec.fromDoubles(xs)
    val s = Vec.softmax(v)
    val y = s(0) * 1.0 + s(1) * 2.0 + s(2) * 3.0
    val g = Value.gradients(y)
    for i <- xs.indices do
      val h = 1e-5
      val num = (f(xs.updated(i, xs(i) + h)) - f(xs.updated(i, xs(i) - h))) / (2 * h)
      assertEqualsDouble(g(v(i)), num, 1e-5)
  }

  test("layerNorm 後は平均 0・分散 1 に近い") {
    val v = Vec.fromDoubles(Seq(1.0, 2.0, 3.0, 4.0))
    val n = Vec.data(Vec.layerNorm(v))
    val mean = n.sum / n.size
    val variance = n.map(x => (x - mean) * (x - mean)).sum / n.size
    assertEqualsDouble(mean, 0.0, 1e-9)
    assertEqualsDouble(variance, 1.0, 1e-3)
  }

  test("relu は負を 0 にする") {
    val v = Vec.fromDoubles(Seq(-1.0, 0.5))
    assertEquals(Vec.data(Vec.relu(v)), Vector(0.0, 0.5))
  }

  test("mean と concat") {
    val a = Vec.fromDoubles(Seq(1.0, 3.0))
    val b = Vec.fromDoubles(Seq(5.0))
    assertEqualsDouble(Vec.mean(a).data, 2.0, eps)
    assertEquals(Vec.data(Vec.concat(a, b)), Vector(1.0, 3.0, 5.0))
  }
