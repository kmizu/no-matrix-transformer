# 付録 傾きを測ってつなぐ：逆伝播の正体

本線は「全部を一斉にゆらして、損失を見比べる」で学習しました。届く場所は逆伝播と同じでしたが、15 倍の時間がかかりました。
この付録では、同じモデルを**逆伝播**で学習します。第14章で書いたとおり、逆伝播が速いのは微分だからではなく、
「計算の途中の数を全部使う」からです。ここでは、それを**微分の公式を一つも使わずに**、行列もなしで実装します。

使う事実は 3 つだけです。

1. 各部品の傾き（入力を少し動かすと出力がどれだけ動くか）は、**その場で動かして測れる**
2. 経路に沿って、傾きは**掛け算**になる（A が B を 3 倍で動かし、B が C を 2 倍で動かすなら、A は C を 6 倍で動かす）
3. 複数の経路で届くなら、**足し算**になる

付録のコードは `nomatrix.backprop` パッケージにまとめてあり、本線のコードは一切触っていません。
同じパラメータを渡せば本線と同じ損失を出すことを、`EquivalenceTest` が 1e-9 の精度で確かめています。

## 考え方: 傾きを持ち歩く数

本線の数は `Double` でした。「少し動かしたら損失がどう変わるか」を知るには、モデル全体を動かして測るしかありません。
7,000 個の数について、それぞれ。

ここでは、**計算しながら「自分がどこからどれだけの傾きで作られたか」を記録する数**を作ります。これが `Value` です。
傾きは、その部品の入力をわずかに動かして測ります。

```scala
/** x を h だけ前後に動かして f の変化を見る。傾き = 変化 ÷ 動かした幅。 */
def measure(f: Double => Double, x: Double): Double =
  val h = 1e-6 * math.max(1.0, math.abs(x))
  (f(x + h) - f(x - h)) / (2 * h)
```

たとえば \( c = a \times b \) なら、\( a \) を少し動かして \( c \) の変化を測ると、\( b \) 倍動くと分かります。
この「局所的な傾き」は掛け算した時点で測れるので、`c` にそのまま記録しておきます。
本線の「動かして測る」を、モデル全体ではなく**部品ひとつぶんだけ**やっているわけです。

```scala mdoc
import nomatrix.backprop.Value

val a = Value(2.0)
val b = Value(3.0)
val c = a * b
c.inputs.map((in, slope) => (in.data, slope))
```

`c` は「`a`（値 2.0）から傾き 3.0 で、`b`（値 3.0）から傾き 2.0 で作られた」と覚えています。

## 逆伝播: 傾きを後ろから配る

最終的な結果（損失）から出発して、記録された傾きを掛けながら入力側へ配っていくと、
すべてのノードについて「損失をそのノードで微分した値」が求まります。これが**逆伝播**、
数学でいう**連鎖律**です。

```scala mdoc
val y = a * b + a          // y = a*b + a
val grads = Value.gradients(y)
grads(a)                   // dy/da = b + 1 = 4
grads(b)                   // dy/db = a = 2
```

`a` は 2 回使われています。逆伝播は、両方の経路から来た傾きを足し合わせます。

## 全体を動かして測るのと一致する

本線の方法（モデル全体を少し動かして損失の差を取る）と、部品ごとに測った傾きをつないだ答えが一致することを確かめます。

```scala mdoc
def f(x: Double): Double = math.tanh(x * x + math.exp(x))
def fv(x: Value): Value = (x * x + x.exp).tanh

val x0 = 0.3
val h = 1e-5
val nudged = (f(x0 + h) - f(x0 - h)) / (2 * h)   // 本線の考え方: 動かして測る

val x = Value(x0)
val auto = Value.gradients(fv(x))(x)               // 逆伝播: 記録から計算する

math.abs(nudged - auto) < 1e-6
```

同じ答えです。違いは、本線が「動かした方向 1 本ぶん」しか分からないのに対して、
こちらは 1 回の計算で「すべての入力について」の傾きが出ることです。
部品の傾きを測るのに要るのは、その部品の評価 2 回だけ。モデル全体を動かす必要がありません。

## 実装を読む

`Value` の全体です。行列はもちろん、配列すら使っていません。微分の公式も出てきません。

```scala
--8<-- "src/main/scala/nomatrix/backprop/Value.scala"
```

読みどころは 3 つです。

**1. ノードは不変（immutable）**
`data` と `inputs` はコンストラクタで確定し、以後変わりません。
逆伝播の結果は `Gradients` という別のオブジェクトに入ります。

**2. `Value.sum` はひとつのノード**
1000 個の数を `+` で順に足すと、深さ 1000 のグラフになります。
`sum` は「入力が 1000 個あるノードひとつ」にするので、グラフが浅く保たれます。

**3. 再帰を使わない**
`topologicalOrder` は明示的なスタックで深さ優先探索します。
Transformer のグラフは数万ノードになるので、再帰だとスタックがあふれます。

## 同じ部品を `Value` で書き直す

本線の `Vec`、`Dense`、`AttentionHead` などを、中身の数だけ `Double` から `Value` に替えたものが
`nomatrix.backprop` にあります。構造は本線とまったく同じです。
`Dense` を並べて見比べると、違いは `Double` が `Value` になっていることだけだと分かります。

```scala
--8<-- "src/main/scala/nomatrix/backprop/Dense.scala"
```

パラメータの辞書 `Params` は本線と共有です。順伝播の入口で `lift` して「微分できる数」に持ち上げ、
逆伝播の結果を名前で受け取ります。

```scala mdoc
import nomatrix.backprop.*
import nomatrix.nn.Params
import nomatrix.vec

val params = nomatrix.nn.Dense.init("layer", in = 3, out = 2, new scala.util.Random(0))
val pv = params.lift
val dense = Dense.load(pv, "layer", in = 3, out = 2)
val out = dense(GVec.fromDoubles(Seq(1.0, 0.0, -1.0)))
val g = pv.gradients(Value.gradients(Value.sum(out)))
g("layer.n0.b")     // バイアスの勾配は 1
g("layer.n0.w2")    // 2 番目の重みの勾配 = 入力の 2 番目 = -1.0
```

第3章で「名前で動かす」と確かめたのと同じことが、動かさずに一度に全部求まっています。

## 損失の勾配は「確率 − 正解」

交差エントロピーを逆伝播すると、きれいな形になります。
ロジット \( j \) に対する勾配は、\( p_j - [j = \text{正解}] \) です。

```scala mdoc
val logits = GVec.fromDoubles(Seq(0.5, -1.0, 2.0))
val probs = GVec.data(GVec.softmax(logits))
val lg = Value.gradients(Loss.crossEntropy(logits, 0))
logits.map(lg(_))
probs.map(p => f"$p%.3f")
```

正解（0 番）の勾配は「確率 − 1」、不正解は確率そのもの。
「正解のスコアを上げ、不正解のスコアを（確率に比例して）下げる」方向です。

## 勾配降下と Adam

勾配が分かれば、その逆向きに少し動くだけです。本線と同じく Adam で歩幅を整えます。
本線では「ゆらぎで推定した向き」を渡していたところに、「正確な勾配」を渡すだけの違いです。

```scala mdoc
val start = Params(Map("x" -> 5.0, "y" -> -3.0))

// loss = x^2 + (y - 1)^2 を最小化する。答えは x = 0, y = 1
val (end, _) = (1 to 300).foldLeft((start, AdamState.initial)) { case ((p, s), _) =>
  val pv = p.lift
  val loss = pv("x") * pv("x") + (pv("y") - 1.0) * (pv("y") - 1.0)
  Adam.step(p, pv.gradients(Value.gradients(loss)), s, lr = 0.1)
}
end.values.map((k, v) => k -> f"$v%.3f")
```

## 学習ループ

```scala
val (params, _, log) = (1 to steps).foldLeft(start) { case ((p, state, log), step) =>
  val window = sampleWindow(corpus, cfg.context, rng)
  val (loss, grads) = lossAndGradients(cfg, p, window)
  val (next, nextState) = Adam.step(p, grads, state, lr)
  (next, nextState, log :+ TrainStep(step, loss))
}
```

本線のループと形は同じです。「ゆらぎを何組か試す」が「逆伝播 1 回」に置き換わっています。

## このページの上で 150 ステップ学習する

以下のコードは、このサイトを生成するたびに実際に実行されています。CPU で十数秒です。

```scala mdoc
import nomatrix.data.Corpus

val tok = Corpus.tokenizer
val cfg = Corpus.defaultConfig(tok.vocabSize)
val rng = new scala.util.Random(42)
val (after150, log) = Trainer.train(cfg, nomatrix.model.Transformer.init(cfg, rng), Corpus.ids, steps = 150, lr = 0.01, rng = rng)

log.filter(_.step % 25 == 0).map(s => f"step ${s.step}%3d  loss ${s.loss}%.3f")
```

本線のゆらぎ学習では、同じ 150 ステップではここまで下がりません。
1 ステップの情報量が違うからです。

## 2000 ステップ学習したモデル

```bash
sbt "runMain nomatrix.Main train --backprop --steps 2000 --lr 0.01 --seed 7"
```

手元（CPU、1 コア）で約 4 分。学習ログの抜粋です。

--8<-- "docs/site/snippets/backprop2000.md"

このパラメータは `src/main/resources/pretrained-backprop.txt` に同梱してあり、`Pretrained.backprop` で読めます。
本線の学習済みモデルと同じプロンプトで生成して見比べます。

```scala mdoc
import nomatrix.data.Pretrained
import nomatrix.gen.Generator

def write(p: Params, prompt: String, seed: Int): String =
  tok.decode(Generator.generate(cfg, p, tok.encode(prompt), count = 60, temperature = 0.5, rng = new scala.util.Random(seed)))

write(Pretrained.backprop, "あさ", 1)
write(Pretrained.params, "あさ", 1)
```

上が逆伝播、下がゆらぎ学習です。

## 本線と同じ答えを出すこと

同じパラメータ、同じ窓で、本線（`Double`）と付録（`Value`）の損失を比べます。

```scala mdoc
val window = Corpus.ids.take(cfg.context + 1)
val (viaValue, _) = Trainer.lossAndGradients(cfg, Pretrained.params, window)
val viaDouble = nomatrix.train.Trainer.loss(cfg, Pretrained.params, window)
math.abs(viaValue - viaDouble) < 1e-9
```

`Value` は「答えが同じで、ついでに傾きも分かる数」です。答えを変えるものではありません。

!!! tip "この付録のまとめ"
    - `Value` = 値 + 「誰からどれだけの傾きで作られたか」。傾きは部品ごとに動かして測る
    - 経路に沿って掛け、経路ごとに足す。それだけで全パラメータの傾きが一度に求まる
    - 部品の構造は本線と同じ。数の種類が違うだけ。答えも同じで、違うのは速さ
    - 世間で「微分」「連鎖律」「逆伝播」と呼ばれているものの正体は、この「測って、掛けて、足す」
