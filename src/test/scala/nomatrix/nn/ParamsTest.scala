package nomatrix.nn

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

  test("updated は名前の数だけを変え、存在しない名前は例外") {
    val p = Params(Map("w" -> 3.0, "b" -> 1.0))
    val q = p.updated("w", 5.0)
    assertEquals(q("w"), 5.0)
    assertEquals(q("b"), 1.0)
    assertEquals(p("w"), 3.0)
    intercept[IllegalArgumentException](p.updated("nope", 1.0))
    intercept[NoSuchElementException](p("nope"))
  }

  test("save / parse で往復できる") {
    val p = Params(Map("a.w0" -> 0.5, "b" -> -1.25))
    val text = p.toText
    assertEquals(Params.parse(text), p)
  }
