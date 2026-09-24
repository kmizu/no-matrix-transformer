package nomatrix

import nomatrix.data.Corpus
import java.nio.file.Files

class MainTest extends munit.FunSuite:

  test("引数の解析") {
    assertEquals(Main.parse(List("train", "--steps", "5", "--lr", "0.1")).map(o => (o.steps, o.lr)), Right((5, 0.1)))
    assertEquals(Main.parse(List("generate", "--prompt", "よる", "--count", "3", "--temperature", "0", "--seed", "9", "--params", "p.txt"))
      .map(o => (o.prompt, o.count, o.temperature, o.seed, o.paramsPath)), Right(("よる", 3, 0.0, 9L, "p.txt")))
    assertEquals(Main.parse(List("train", "--backprop", "--pairs", "8", "--sigma", "0.1", "--windows", "2"))
      .map(o => (o.backprop, o.pairs, o.sigma, o.windows)), Right((true, 8, 0.1, 2)))
    assert(Main.parse(List("dance")).isLeft)
    assert(Main.parse(List("train", "--steps", "x")).isLeft)
    assert(Main.parse(List("train", "--bogus")).isLeft)
  }

  test("同梱コーパスはひらがな中心で、語彙は小さい") {
    assert(Corpus.tokenizer.vocabSize < 80)
    assert(Corpus.ids.length > 500)
  }

  test("train → generate が通しで動く（極小ステップ）") {
    val file = Files.createTempFile("params", ".txt")
    val lines = Vector.newBuilder[String]
    Main.run(Main.Options("train", steps = 2, pairs = 4, windows = 1, paramsPath = file.toString), lines += _)
    assert(lines.result().exists(_.startsWith("保存")))
    val bp = Vector.newBuilder[String]
    Main.run(Main.Options("train", steps = 1, backprop = true, paramsPath = file.toString), bp += _)
    assert(bp.result().exists(_.contains("逆伝播")))
    val gen = Vector.newBuilder[String]
    Main.run(Main.Options("generate", paramsPath = file.toString, count = 3, prompt = "あさ"), gen += _)
    val text = gen.result().head
    assert(text.startsWith("あさ"))
    assertEquals(text.length, 5)
  }
