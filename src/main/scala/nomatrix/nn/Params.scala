package nomatrix.nn

/** 学習するパラメータ = 「名前付きの数」の集まり。
  *
  * 行列でも配列でもなく、ただの辞書。`"block0.attn.head1.query.n3.w5"` のような名前に数がひとつ付く。
  */
final case class Params(values: Map[String, Double]):

  def apply(name: String): Double = values(name)

  def size: Int = values.size

  def names: Iterable[String] = values.keys

  /** 名前の衝突を許さない結合。 */
  def ++(other: Params): Params =
    val dup = values.keySet.intersect(other.values.keySet)
    require(dup.isEmpty, s"パラメータ名が重複: ${dup.take(3).mkString(", ")}")
    Params(values ++ other.values)

  /** 名前ごとに数を変える。 */
  def updated(name: String, value: Double): Params =
    require(values.contains(name), s"パラメータが無い: $name")
    Params(values.updated(name, value))

  /** 1 行 1 パラメータのテキスト。 */
  def toText: String =
    values.toVector.sortBy(_._1).map((n, d) => s"$n ${d.toString}").mkString("\n")

object Params:
  val empty: Params = Params(Map.empty)

  def parse(text: String): Params =
    val entries = text.linesIterator.filter(_.trim.nonEmpty).map { line =>
      line.trim.split(" ", 2) match
        case Array(n, d) => n -> d.toDouble
        case _           => throw new IllegalArgumentException(s"読めない行: $line")
    }
    Params(entries.toMap)
