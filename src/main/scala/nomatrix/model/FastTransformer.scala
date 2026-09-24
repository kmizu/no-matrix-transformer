package nomatrix.model

import nomatrix.nn.Params
import scala.util.Random

/** `Plain` と同じ計算を、名前引きを一度だけにして配列で行う版。ゆらぎ学習は順伝播を何万回も呼ぶので速さが要る。 */
final class FastTransformer(val cfg: Config):

  /** 名前 → 配列上の位置。名前はソート順で固定。 */
  val names: Vector[String] = Transformer.init(cfg, new Random(0)).names.toVector.sorted
  private val index: Map[String, Int] = names.zipWithIndex.toMap
  val size: Int = names.length

  def toArray(p: Params): Array[Double] = names.map(p(_)).toArray
  def toParams(a: Array[Double]): Params = Params(names.zip(a).toMap)

  private def idx(name: String): Int = index(name)

  private final class DenseIx(prefix: String, in: Int, out: Int):
    val w: Vector[Array[Int]] = Vector.tabulate(out)(j => Array.tabulate(in)(i => idx(s"$prefix.n$j.w$i")))
    val b: Array[Int] = Array.tabulate(out)(j => idx(s"$prefix.n$j.b"))
    def apply(p: Array[Double], x: Array[Double]): Array[Double] =
      val y = new Array[Double](out)
      var j = 0
      while j < out do
        var s = p(b(j))
        val row = w(j)
        var i = 0
        while i < in do
          s += p(row(i)) * x(i)
          i += 1
        y(j) = s
        j += 1
      y

  private final class EmbIx(prefix: String, count: Int, dim: Int):
    val t: Vector[Array[Int]] = Vector.tabulate(count)(t => Array.tabulate(dim)(d => idx(s"$prefix.t$t.d$d")))
    def apply(p: Array[Double], id: Int): Array[Double] = t(id).map(p(_))

  private final class NormIx(prefix: String, dim: Int):
    val g: Array[Int] = Array.tabulate(dim)(d => idx(s"$prefix.g$d"))
    val b: Array[Int] = Array.tabulate(dim)(d => idx(s"$prefix.b$d"))
    def apply(p: Array[Double], x: Array[Double]): Array[Double] =
      var m = 0.0
      x.foreach(m += _)
      m /= dim
      var v = 0.0
      x.foreach(e => v += (e - m) * (e - m))
      v /= dim
      val inv = 1.0 / math.sqrt(v + 1e-5)
      Array.tabulate(dim)(d => (x(d) - m) * inv * p(g(d)) + p(b(d)))

  private val headDim = cfg.dModel / cfg.heads
  private val tok = EmbIx("tok", cfg.vocabSize, cfg.dModel)
  private val pos = EmbIx("pos", cfg.context, cfg.dModel)
  private final case class BlockIx(norm1: NormIx, heads: Vector[(DenseIx, DenseIx, DenseIx)], output: DenseIx, norm2: NormIx, up: DenseIx, down: DenseIx)
  private val blocks = Vector.tabulate(cfg.layers) { i =>
    val b = s"block$i"
    BlockIx(
      NormIx(s"$b.norm1", cfg.dModel),
      Vector.tabulate(cfg.heads) { h =>
        val hp = s"$b.attn.head$h"
        (DenseIx(s"$hp.query", cfg.dModel, headDim), DenseIx(s"$hp.key", cfg.dModel, headDim), DenseIx(s"$hp.value", cfg.dModel, headDim))
      },
      DenseIx(s"$b.attn.output", cfg.dModel, cfg.dModel),
      NormIx(s"$b.norm2", cfg.dModel),
      DenseIx(s"$b.ff.up", cfg.dModel, cfg.hidden),
      DenseIx(s"$b.ff.down", cfg.hidden, cfg.dModel))
  }
  private val norm = NormIx("norm", cfg.dModel)
  private val head = DenseIx("head", cfg.dModel, cfg.vocabSize)

  private def dot(a: Array[Double], b: Array[Double]): Double =
    var s = 0.0
    var i = 0
    while i < a.length do
      s += a(i) * b(i)
      i += 1
    s

  private def attention(p: Array[Double], q: DenseIx, k: DenseIx, v: DenseIx, xs: Vector[Array[Double]]): Vector[Array[Double]] =
    val qs = xs.map(q(p, _))
    val ks = xs.map(k(p, _))
    val vs = xs.map(v(p, _))
    val scale = 1.0 / math.sqrt(headDim.toDouble)
    Vector.tabulate(xs.length) { i =>
      val scores = Array.tabulate(i + 1)(j => dot(qs(i), ks(j)) * scale)
      val shift = scores.max
      val exps = scores.map(s => math.exp(s - shift))
      val total = exps.sum
      val out = new Array[Double](headDim)
      var j = 0
      while j <= i do
        val w = exps(j) / total
        var d = 0
        while d < headDim do
          out(d) += w * vs(j)(d)
          d += 1
        j += 1
      out
    }

  private def runBlock(p: Array[Double], b: BlockIx, xs: Vector[Array[Double]]): Vector[Array[Double]] =
    val normed = xs.map(b.norm1(p, _))
    val perHead = b.heads.map((q, k, v) => attention(p, q, k, v, normed))
    val mixed = Vector.tabulate(xs.length)(i => b.output(p, perHead.map(_(i)).reduce(_ ++ _)))
    val afterAttention = xs.zip(mixed).map((x, m) => x.zip(m).map(_ + _))
    afterAttention.map { x =>
      val ff = b.down(p, b.up(p, b.norm2(p, x)).map(v => math.max(0.0, v)))
      x.zip(ff).map(_ + _)
    }

  /** 埋め込みから `upTo` 個のブロックを通したあとの、各トークンの状態。 */
  def hiddenAfter(p: Array[Double], ids: Vector[Int], upTo: Int): Vector[Array[Double]] =
    val embedded = ids.zipWithIndex.map((id, i) => tok(p, id).zip(pos(p, i)).map(_ + _))
    blocks.take(upTo).foldLeft(embedded)((xs, b) => runBlock(p, b, xs))

  /** 途中の状態 `hidden`（ブロック `from` の入力）から最後まで通して、損失を出す。 */
  def lossFrom(p: Array[Double], hidden: Vector[Array[Double]], from: Int, targets: Vector[Int]): Double =
    val xs = blocks.drop(from).foldLeft(hidden)((h, b) => runBlock(p, b, h))
    lossOfLogits(xs.map(x => head(p, norm(p, x))), targets)

  /** 途中の状態にそのまま仕上げ（LayerNorm と head）を当てて損失を出す。ブロックごとの局所損失に使う。 */
  def readoutLoss(p: Array[Double], hidden: Vector[Array[Double]], targets: Vector[Int]): Double =
    lossOfLogits(hidden.map(x => head(p, norm(p, x))), targets)

  def logits(p: Array[Double], ids: Vector[Int]): Vector[Array[Double]] =
    hiddenAfter(p, ids, blocks.length).map(x => head(p, norm(p, x)))

  private def lossOfLogits(out: Vector[Array[Double]], targets: Vector[Int]): Double =
    var total = 0.0
    for (l, t) <- out.zip(targets) do
      val shift = l.max
      total += math.log(l.map(x => math.exp(x - shift)).sum) + shift - l(t)
    total / targets.length

  def loss(p: Array[Double], window: Vector[Int]): Double =
    lossOfLogits(logits(p, window.dropRight(1)), window.drop(1))
