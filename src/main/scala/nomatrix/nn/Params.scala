package nomatrix.nn

import nomatrix.autograd.{Gradients, Value}

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

  /** 順伝播の入口で、数を「微分できる数」に持ち上げる。 */
  def lift: ParamValues = ParamValues(values.map((n, d) => n -> Value.leaf(d, n)))

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

/** 持ち上げ済みのパラメータ。順伝播中はこちらを引く。 */
final case class ParamValues(values: Map[String, Value]):

  def apply(name: String): Value =
    values.getOrElse(name, throw new NoSuchElementException(s"パラメータが無い: $name"))

  /** 逆伝播の結果を「名前 → 勾配」に読み替える。 */
  def gradients(g: Gradients): Map[String, Double] =
    values.map((n, v) => n -> g(v))
