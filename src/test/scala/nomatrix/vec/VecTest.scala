package nomatrix.vec

class VecTest extends munit.FunSuite:
  private val eps = 1e-9

  test("add / sub / mul / scale は要素ごと") {
    val a = Vector(1.0, 2.0)
    val b = Vector(10.0, 20.0)
    assertEquals(Vec.add(a, b), Vector(11.0, 22.0))
    assertEquals(Vec.sub(b, a), Vector(9.0, 18.0))
    assertEquals(Vec.mul(a, b), Vector(10.0, 40.0))
    assertEquals(Vec.scale(a, 3.0), Vector(3.0, 6.0))
  }

  test("長さが違うベクトルの add は例外") {
    intercept[IllegalArgumentException](Vec.add(Vector(1.0), Vector(1.0, 2.0)))
  }

  test("dot は積の和") {
    assertEqualsDouble(Vec.dot(Vector(1.0, 2.0, 3.0), Vector(4.0, 5.0, 6.0)), 32.0, eps)
  }

  test("softmax は合計 1 で、大きい値ほど大きい割合") {
    val s = Vec.softmax(Vector(1.0, 2.0, 3.0))
    assertEqualsDouble(s.sum, 1.0, eps)
    assert(s(2) > s(1) && s(1) > s(0))
  }

  test("softmax は大きな値でもオーバーフローしない") {
    val s = Vec.softmax(Vector(1000.0, 1000.0))
    assertEqualsDouble(s(0), 0.5, eps)
  }

  test("logSumExp は log(Σ exp) と一致し、大きな値でも壊れない") {
    val v = Vector(0.5, -1.0, 2.0)
    assertEqualsDouble(Vec.logSumExp(v), math.log(v.map(math.exp).sum), eps)
    assert(!Vec.logSumExp(Vector(1000.0, -1000.0)).isNaN)
  }

  test("layerNorm 後は平均 0・分散 1 に近い") {
    val n = Vec.layerNorm(Vector(1.0, 2.0, 3.0, 4.0))
    val mean = n.sum / n.size
    val variance = n.map(x => (x - mean) * (x - mean)).sum / n.size
    assertEqualsDouble(mean, 0.0, 1e-9)
    assertEqualsDouble(variance, 1.0, 1e-3)
  }

  test("relu / mean / concat") {
    assertEquals(Vec.relu(Vector(-1.0, 0.5)), Vector(0.0, 0.5))
    assertEqualsDouble(Vec.mean(Vector(1.0, 3.0)), 2.0, eps)
    assertEquals(Vec.concat(Vector(1.0), Vector(2.0, 3.0)), Vector(1.0, 2.0, 3.0))
  }
