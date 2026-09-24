package nomatrix.backprop

/** 傾きを持ち歩く「ひとつの数」。
  *
  * `data` はこのノードの値。`inputs` は「どのノードから、どれだけの傾きで作られたか」の一覧。
  * 傾きは微分の公式ではなく、**その場で入力をわずかに動かして測る**（`Slope.measure`）。
  * 測った傾きは作った時点で確定するので、ノードは完全に不変（immutable）。
  *
  * 行列もベクトルも登場しない。ここにあるのはスカラーと、そのつながりだけ。
  */
final class Value private[backprop] (
    val data: Double,
    val inputs: Vector[(Value, Double)],
    val label: String
):

  /** 1 入力の部品: 値を計算し、入力を動かしたときの傾きを測る。 */
  private def unary(f: Double => Double): Value =
    Value.node(f(data), Vector(this -> Slope.measure(f, data)))

  /** 2 入力の部品: それぞれの入力について傾きを測る。 */
  private def binary(o: Value, f: (Double, Double) => Double): Value =
    Value.node(f(data, o.data), Vector(this -> Slope.measure(f(_, o.data), data), o -> Slope.measure(f(data, _), o.data)))

  def +(o: Value): Value = binary(o, _ + _)
  def -(o: Value): Value = binary(o, _ - _)
  def *(o: Value): Value = binary(o, _ * _)
  def /(o: Value): Value = binary(o, _ / _)

  def +(d: Double): Value = unary(_ + d)
  def -(d: Double): Value = unary(_ - d)
  def *(d: Double): Value = unary(_ * d)
  def /(d: Double): Value = unary(_ / d)

  def unary_- : Value = this * -1.0

  def pow(k: Double): Value = unary(math.pow(_, k))
  def sqrt: Value = unary(math.sqrt)
  def exp: Value = unary(math.exp)
  def log: Value = unary(math.log)
  def tanh: Value = unary(math.tanh)
  def relu: Value = unary(x => math.max(0.0, x))

  override def toString: String =
    if label.nonEmpty then s"Value($data, $label)" else s"Value($data)"

/** 傾きを測る。公式は使わない。 */
object Slope:

  /** x を h だけ前後に動かして f の変化を見る。傾き = 変化 ÷ 動かした幅。 */
  def measure(f: Double => Double, x: Double): Double =
    val h = 1e-6 * math.max(1.0, math.abs(x))
    (f(x + h) - f(x - h)) / (2 * h)

object Value:

  /** 名前のない葉ノード（定数や入力）。 */
  def apply(d: Double): Value = new Value(d, Vector.empty, "")

  /** 名前付きの葉ノード。学習するパラメータに使う。 */
  def leaf(d: Double, label: String): Value = new Value(d, Vector.empty, label)

  private[backprop] def node(d: Double, inputs: Vector[(Value, Double)]): Value =
    new Value(d, inputs, "")

  /** たくさんの数の合計。どの入力を動かしても合計は同じだけ動くので、傾きは 1。ひとつのノードにまとめるのでグラフが浅く保たれる。 */
  def sum(xs: Iterable[Value]): Value =
    val vec = xs.toVector
    node(vec.foldLeft(0.0)(_ + _.data), vec.map(_ -> 1.0))

  /** 逆伝播。`root` を各ノードで「少し動かしたらどれだけ動くか」をまとめて返す。
    *
    * 使う事実は 2 つだけ: 経路に沿って傾きは掛け算、複数の経路は足し算。
    */
  def gradients(root: Value): Gradients =
    val order = topologicalOrder(root)
    val grads = new java.util.IdentityHashMap[Value, java.lang.Double]()
    grads.put(root, 1.0)
    // 出力側から入力側へ、傾きを掛けながら配っていく
    order.reverseIterator.foreach { v =>
      val g = grads.getOrDefault(v, 0.0).doubleValue
      v.inputs.foreach { (in, local) =>
        grads.put(in, grads.getOrDefault(in, 0.0).doubleValue + g * local)
      }
    }
    new Gradients(grads)

  /** 入力側が先、出力側が後になるようにノードを並べる（再帰なし）。 */
  private def topologicalOrder(root: Value): Vector[Value] =
    val visited = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap[Value, java.lang.Boolean]())
    val out = Vector.newBuilder[Value]
    val stack = scala.collection.mutable.Stack[(Value, Boolean)]((root, false))
    while stack.nonEmpty do
      val (v, expanded) = stack.pop()
      if expanded then out += v
      else if !visited.contains(v) then
        visited.add(v)
        stack.push((v, true))
        v.inputs.foreach { (in, _) => if !visited.contains(in) then stack.push((in, false)) }
    out.result()

/** 逆伝播の結果。ノードを渡すと、そのノードを少し動かしたときに root がどれだけ動くかを返す。 */
final class Gradients private[backprop] (private val map: java.util.IdentityHashMap[Value, java.lang.Double]):
  def apply(v: Value): Double = map.getOrDefault(v, 0.0).doubleValue
