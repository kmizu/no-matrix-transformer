package nomatrix.data

/** 文字単位のトークナイザ。文字 ↔ 番号 の対応表があるだけ。 */
final case class Tokenizer(chars: Vector[Char]):

  private val toId: Map[Char, Int] = chars.zipWithIndex.toMap

  def vocabSize: Int = chars.length

  def encode(text: String): Vector[Int] =
    text.toVector.map(c => toId.getOrElse(c, throw new IllegalArgumentException(s"未知の文字: '$c'")))

  def decode(ids: Seq[Int]): String = ids.map(chars(_)).mkString

object Tokenizer:
  def fromText(text: String): Tokenizer = Tokenizer(text.toVector.distinct.sorted)
