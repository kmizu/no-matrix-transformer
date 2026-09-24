package nomatrix.backprop

/** 微分できる「ひとつの数」。
  *
  * `data` はこのノードの値。`inputs` は「どのノードから、どれだけの傾き（局所微分）で
  * 作られたか」の一覧。局所微分は順伝播の時点で確定するので、ノードは完全に不変（immutable）。
  *
  * 行列もベクトルも登場しない。ここにあるのはスカラーと、そのつながりだけ。
  */
final class Value private[backprop] (
    val data: Double,
    val inputs: Vector[(Value, Double)],
    val label: String
):

  def +(o: Value): Value = Value.node(data + o.data, Vector(this -> 1.0, o -> 1.0))
  def -(o: Value): Value = Value.node(data - o.data, Vector(this -> 1.0, o -> -1.0))
  def *(o: Value): Value = Value.node(data * o.data, Vector(this -> o.data, o -> data))
  def /(o: Value): Value = this * o.pow(-1.0)

  def +(d: Double): Value = Value.node(data + d, Vector(this -> 1.0))
  def -(d: Double): Value = Value.node(data - d, Vector(this -> 1.0))
  def *(d: Double): Value = Value.node(data * d, Vector(this -> d))
  def /(d: Double): Value = Value.node(data / d, Vector(this -> 1.0 / d))

  def unary_- : Value = this * -1.0

  def pow(k: Double): Value =
    Value.node(math.pow(data, k), Vector(this -> k * math.pow(data, k - 1)))

  def sqrt: Value = pow(0.5)

  def exp: Value =
    val e = math.exp(data)
    Value.node(e, Vector(this -> e))

  def log: Value = Value.node(math.log(data), Vector(this -> 1.0 / data))

  def tanh: Value =
    val t = math.tanh(data)
    Value.node(t, Vector(this -> (1.0 - t * t)))

  def relu: Value =
    if data > 0 then Value.node(data, Vector(this -> 1.0))
    else Value.node(0.0, Vector(this -> 0.0))

  override def toString: String =
    if label.nonEmpty then s"Value($data, $label)" else s"Value($data)"

object Value:

  /** 名前のない葉ノード（定数や入力）。 */
  def apply(d: Double): Value = new Value(d, Vector.empty, "")

  /** 名前付きの葉ノード。学習するパラメータに使う。 */
  def leaf(d: Double, label: String): Value = new Value(d, Vector.empty, label)

  private[backprop] def node(d: Double, inputs: Vector[(Value, Double)]): Value =
    new Value(d, inputs, "")

  /** たくさんの数の合計。ひとつのノードにまとめるのでグラフが浅く保たれる。 */
  def sum(xs: Iterable[Value]): Value =
    val vec = xs.toVector
    node(vec.foldLeft(0.0)(_ + _.data), vec.map(_ -> 1.0))

  /** 逆伝播。`root` を各ノードで微分した値をまとめて返す。 */
  def gradients(root: Value): Gradients =
    val order = topologicalOrder(root)
    val grads = new java.util.IdentityHashMap[Value, java.lang.Double]()
    grads.put(root, 1.0)
    // 出力側から入力側へ、傾きを掛けながら配っていく（連鎖律）
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

/** 逆伝播の結果。ノードを渡すと、そのノードに対する微分を返す。 */
final class Gradients private[backprop] (private val map: java.util.IdentityHashMap[Value, java.lang.Double]):
  def apply(v: Value): Double = map.getOrDefault(v, 0.0).doubleValue
