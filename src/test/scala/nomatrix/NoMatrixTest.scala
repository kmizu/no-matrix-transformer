package nomatrix

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

/** このプロジェクトの約束: 行列型も行列積も使わない。ソースを機械的に検査する。 */
class NoMatrixTest extends munit.FunSuite:

  private val forbidden = List("Array[Array", "Vector[Vector[Double]]", "matmul", "Matrix", "行列積")

  test("main ソースに行列型・行列積が登場しない") {
    val root = Path.of("src/main/scala")
    val sources = Files.walk(root).iterator().asScala.filter(_.toString.endsWith(".scala")).toList
    assert(sources.nonEmpty)
    val hits = for
      file <- sources
      line <- Files.readAllLines(file).asScala
      word <- forbidden
      if line.contains(word) && !line.trim.startsWith("//") && !line.trim.startsWith("*")
    yield s"$file: $line"
    assertEquals(hits, Nil)
  }
