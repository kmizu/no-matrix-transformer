package nomatrix.data

class TokenizerTest extends munit.FunSuite:

  test("文字集合はテキストから作られ、ソートされて重複がない") {
    val t = Tokenizer.fromText("ばなな りんご")
    assertEquals(t.vocabSize, t.chars.distinct.size)
    assertEquals(t.chars, t.chars.sorted)
  }

  test("encode / decode で往復できる") {
    val t = Tokenizer.fromText("あいうえお かきくけこ")
    val ids = t.encode("かえこ")
    assertEquals(ids.length, 3)
    assertEquals(t.decode(ids), "かえこ")
  }

  test("未知の文字は例外") {
    val t = Tokenizer.fromText("abc")
    intercept[IllegalArgumentException](t.encode("z"))
  }
