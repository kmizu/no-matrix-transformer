package nomatrix.nn

import nomatrix.autograd.Value

class ParamsTest extends munit.FunSuite:

  test("Params は名前付きの数の集まりで、結合できる") {
    val a = Params(Map("x" -> 1.0))
    val b = Params(Map("y" -> 2.0))
    val c = a ++ b
    assertEquals(c.size, 2)
    assertEquals(c("y"), 2.0)
  }

  test("同じ名前を二重に定義すると例外") {
    intercept[IllegalArgumentException](Params(Map("x" -> 1.0)) ++ Params(Map("x" -> 2.0)))
  }

  test("lift すると名前付きの葉 Value になり、勾配を名前で取り出せる") {
    val p = Params(Map("w" -> 3.0, "b" -> 1.0, "unused" -> 9.0))
    val pv = p.lift
    assertEquals(pv("w").label, "w")
    val y = pv("w") * 2.0 + pv("b")
    val grads = pv.gradients(Value.gradients(y))
    assertEqualsDouble(grads("w"), 2.0, 1e-9)
    assertEqualsDouble(grads("b"), 1.0, 1e-9)
    assertEqualsDouble(grads("unused"), 0.0, 1e-9)
  }

  test("存在しない名前は例外") {
    intercept[NoSuchElementException](Params(Map.empty).lift("nope"))
  }

  test("save / parse で往復できる") {
    val p = Params(Map("a.w0" -> 0.5, "b" -> -1.25))
    val text = p.toText
    assertEquals(Params.parse(text), p)
  }
