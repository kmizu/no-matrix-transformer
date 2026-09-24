# 第4章 ニューロンと Dense

ここが、普通の解説なら「線形層 \( y = Wx + b \)」と行列で書かれる場所です。
この本では行列を使いません。代わりに**ニューロン**という言葉で書きます。

## ニューロン = 重み付き和 + バイアス

ニューロンは「入力の数の並びを受け取って、ひとつの数を返す関数」です。
自分の重み（入力と同じ長さの数の並び）との内積を取り、バイアスを足すだけ。

\[
\text{neuron}(x) = \sum_i w_i \, x_i + b
\]

```scala mdoc
import nomatrix.autograd.Value
import nomatrix.vec.Vec
import nomatrix.nn.*

val neuron = Neuron(weights = Vec.fromDoubles(Seq(1.0, -2.0, 0.5)), bias = Value(0.1))
neuron(Vec.fromDoubles(Seq(2.0, 1.0, 4.0))).data   // 1*2 + (-2)*1 + 0.5*4 + 0.1
```

## Dense = ニューロンの集まり

入力ベクトルを、たくさんのニューロンに**同時に見せて**、それぞれの答えを並べたものが `Dense` です。
出力の長さは、ニューロンの数です。

```scala
final case class Dense(neurons: Vector[Neuron]):
  def apply(x: Vec): Vec = neurons.map(_(x))
```

これで終わりです。世間で「行列とベクトルの積」と呼ばれているものの正体は、
「ニューロンが並んでいて、それぞれが内積を取る」ことです。

```mermaid
flowchart LR
  x["入力 x<br/>(3 個の数)"] --> n0["ニューロン 0<br/>dot(w0, x) + b0"]
  x --> n1["ニューロン 1<br/>dot(w1, x) + b1"]
  x --> n2["ニューロン 2<br/>dot(w2, x) + b2"]
  x --> n3["ニューロン 3<br/>dot(w3, x) + b3"]
  n0 --> y["出力 y<br/>(4 個の数)"]
  n1 --> y
  n2 --> y
  n3 --> y
```

## パラメータは「名前付きの数」

ニューロンの重みは学習で変わります。学習の対象になる数を、この本では `Params` にまとめます。
`Params` の正体は `Map[String, Double]`。名前がついた数の辞書です。

```scala mdoc
import scala.util.Random

val params = Dense.init("layer", in = 3, out = 2, new Random(0))
params.size
params.names.toVector.sorted
```

`layer.n1.w2` は「`layer` という Dense の、1 番目のニューロンの、2 番目の重み」です。
行列の「1 行 2 列」ではなく、**誰の何番目の重みか**が名前で分かります。

順伝播のときは、この辞書の数を `Value` に持ち上げて（`lift`）、`Dense.load` でニューロンに組み立てます。

```scala mdoc
val pv = params.lift
val dense = Dense.load(pv, "layer", in = 3, out = 2)
val y = dense(Vec.fromDoubles(Seq(1.0, 0.0, -1.0)))
Vec.data(y)
```

## 勾配も名前で受け取る

逆伝播の結果を `ParamValues.gradients` に通すと、「名前 → 勾配」の辞書になります。

```scala mdoc
val loss = Value.sum(y)
val grads = pv.gradients(Value.gradients(loss))
grads("layer.n0.b")     // バイアスの勾配は 1
grads("layer.n0.w0")    // 0 番目の重みの勾配 = 入力の 0 番目 = 1.0
grads("layer.n0.w2")    // 2 番目の重みの勾配 = 入力の 2 番目 = -1.0
```

「重みの勾配は、対応する入力そのもの」という基本が、名前つきで確かめられます。

## 初期値について

重みは `±1/√in` の範囲の一様乱数、バイアスは 0 から始めます。
入力が多いほど重みを小さくして、出力が入力と同じくらいのスケールに収まるようにしています。
これが崩れると、層を重ねたときに数が爆発したり消えたりします。

## 実装を読む

```scala
--8<-- "src/main/scala/nomatrix/nn/Dense.scala"
```

```scala
--8<-- "src/main/scala/nomatrix/nn/Params.scala"
```

!!! tip "この章のまとめ"
    - ニューロン = 内積 + バイアス。Dense = ニューロンの並び
    - 「行列とベクトルの積」は「ニューロンがそれぞれ内積を取る」の別名
    - パラメータは名前付きの数の辞書。勾配も名前で受け取る

次は、文字を数の並びに変える「埋め込み」です。
