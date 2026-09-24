package nomatrix.data

import nomatrix.model.Config
import scala.io.Source
import scala.util.Using

/** 同梱の小さなひらがなコーパス。 */
object Corpus:

  lazy val text: String =
    Using.resource(Source.fromResource("corpus.txt")(using scala.io.Codec.UTF8))(_.mkString)

  lazy val tokenizer: Tokenizer = Tokenizer.fromText(text)

  lazy val ids: Vector[Int] = tokenizer.encode(text)

  /** 手元の CPU で数分で学習できる大きさ。 */
  def defaultConfig(vocabSize: Int): Config =
    Config(vocabSize = vocabSize, dModel = 16, heads = 2, layers = 2, context = 16, hidden = 32)
