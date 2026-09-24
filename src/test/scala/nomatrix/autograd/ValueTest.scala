package nomatrix.autograd

class ValueTest extends munit.FunSuite:

  private val eps = 1e-6

  /** 数値微分: (f(x+h) - f(x-h)) / 2h */
  private def numericGrad(f: Double => Double, x: Double, h: Double = 1e-5): Double =
    (f(x + h) - f(x - h)) / (2 * h)

  test("加算と乗算の値が正しい") {
    val a = Value(2.0)
    val b = Value(3.0)
    assertEqualsDouble((a + b).data, 5.0, eps)
    assertEqualsDouble((a * b).data, 6.0, eps)
    assertEqualsDouble((a - b).data, -1.0, eps)
    assertEqualsDouble((a / b).data, 2.0 / 3.0, eps)
  }

  test("加算の勾配は両方 1") {
    val a = Value(2.0)
    val b = Value(3.0)
    val g = Value.gradients(a + b)
    assertEqualsDouble(g(a), 1.0, eps)
    assertEqualsDouble(g(b), 1.0, eps)
  }

  test("乗算の勾配は相手の値") {
    val a = Value(2.0)
    val b = Value(3.0)
    val g = Value.gradients(a * b)
    assertEqualsDouble(g(a), 3.0, eps)
    assertEqualsDouble(g(b), 2.0, eps)
  }

  test("同じ変数を複数回使うと勾配が足し合わされる") {
    val a = Value(3.0)
    val y = a * a + a // dy/da = 2a + 1 = 7
    assertEqualsDouble(Value.gradients(y)(a), 7.0, eps)
  }

  test("exp / log / tanh / relu / pow の勾配が数値微分と一致する") {
    val fs: List[(String, Value => Value, Double => Double, Double)] = List(
      ("exp", _.exp, math.exp, 0.7),
      ("log", _.log, math.log, 1.3),
      ("tanh", _.tanh, math.tanh, 0.4),
      ("relu+", _.relu, x => math.max(0.0, x), 0.9),
      ("relu-", _.relu, x => math.max(0.0, x), -0.9),
      ("pow3", _.pow(3), x => math.pow(x, 3), 1.1),
      ("sqrt", _.sqrt, math.sqrt, 2.5)
    )
    for (name, fv, fd, x0) <- fs do
      val x = Value(x0)
      val g = Value.gradients(fv(x))(x)
      assertEqualsDouble(g, numericGrad(fd, x0), 1e-5, s"$name")
  }

  test("合成式の勾配が数値微分と一致する") {
    def f(a: Double, b: Double): Double =
      val c = a * b + math.exp(a)
      math.tanh(c) * (a - b) / (b * b + 1.0)
    def fv(a: Value, b: Value): Value =
      val c = a * b + a.exp
      c.tanh * (a - b) / (b * b + 1.0)
    val (a0, b0) = (0.3, -0.8)
    val a = Value(a0)
    val b = Value(b0)
    val g = Value.gradients(fv(a, b))
    assertEqualsDouble(g(a), numericGrad(x => f(x, b0), a0), 1e-5)
    assertEqualsDouble(g(b), numericGrad(x => f(a0, x), b0), 1e-5)
  }

  test("Value.sum は要素それぞれに勾配 1 を流す") {
    val xs = Vector(Value(1.0), Value(2.0), Value(3.0))
    val s = Value.sum(xs)
    assertEqualsDouble(s.data, 6.0, eps)
    val g = Value.gradients(s * 2.0)
    xs.foreach(x => assertEqualsDouble(g(x), 2.0, eps))
  }

  test("深い計算グラフでもスタックオーバーフローしない") {
    val x = Value(1.0)
    val y = (1 to 20000).foldLeft(x)((acc, _) => acc + 1e-3)
    assertEqualsDouble(Value.gradients(y)(x), 1.0, eps)
  }

  test("葉ノードに名前を付けられる") {
    val w = Value.leaf(0.5, "w")
    assertEquals(w.label, "w")
    assertEquals(Value(1.0).label, "")
  }

  test("グラフに含まれないノードの勾配は 0") {
    val a = Value(1.0)
    val other = Value(2.0)
    assertEqualsDouble(Value.gradients(a * 2.0)(other), 0.0, eps)
  }
