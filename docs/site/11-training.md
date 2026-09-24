# 第11章 損失と学習：Adam

モデルは「次の文字のスコア」を出せるようになりました。
でも中身はランダムです。正しい答えに近づける方法が必要です。

## 損失: どれだけ外したか

位置 \( i \) のロジットを softmax で確率にし、**正解の文字の確率**を見ます。
正解の確率が高いほど良い。損失は「正解の確率の対数にマイナスをつけたもの」です。

\[
\text{loss} = -\log p(\text{正解})
\]

正解の確率が 1 なら損失 0。0.5 なら 0.69。0.01 なら 4.6。これを**交差エントロピー**と呼びます。

```scala mdoc
import nomatrix.train.Loss
import nomatrix.vec.Vec

val logits = Vec.fromDoubles(Seq(1.0, 2.0, 3.0))
Vec.data(Vec.softmax(logits))          // 各文字の確率
Loss.crossEntropy(logits, 2).data      // 正解が 2 番なら、-log(0.665)
Loss.crossEntropy(logits, 0).data      // 正解が 0 番なら、-log(0.090)
```

実装では softmax を経由せず、\( \log \sum_j e^{x_j} - x_{\text{正解}} \) と計算しています。
数学的には同じで、こちらの方が桁あふれに強く、グラフも小さくなります。

## 損失の勾配は「確率 − 正解」

交差エントロピーの微分は、きれいな形になります。
ロジット \( j \) に対する勾配は、\( p_j - [j = \text{正解}] \) です。

```scala mdoc
import nomatrix.autograd.Value

val g = Value.gradients(Loss.crossEntropy(logits, 0))
logits.map(g(_))
```

正解（0 番）の勾配は \( 0.090 - 1 = -0.91 \)、不正解は確率そのもの。
「正解のスコアを上げ、不正解のスコアを（確率に比例して）下げる」方向です。

## 文全体の損失

`context` 文字の窓を切り出し、「前 16 文字 → 後ろに 1 つずらした 16 文字」を入力と正解にします。
各位置の損失の平均が、この窓の損失です。

```scala mdoc
import nomatrix.train.Trainer
import scala.util.Random

val corpus = Vector.range(0, 20)
val window = Trainer.sampleWindow(corpus, context = 5, new Random(1))
val inputs = window.dropRight(1)
val targets = window.drop(1)
```

位置 0 は「1 文字だけ見て 2 文字目を当てる」、位置 4 は「5 文字見て 6 文字目を当てる」。
ひとつの窓で context 個（既定では 16 個）の予測問題を同時に解いているわけです。

## 勾配降下: 傾きの逆向きに少し動く

損失をパラメータで微分すると、「このパラメータを増やすと損失がどう変わるか」が分かります。
増えるなら減らし、減るなら増やす。これを全パラメータで一斉にやるのが勾配降下です。

\[
w \leftarrow w - \eta \cdot \frac{\partial \text{loss}}{\partial w}
\]

## Adam: 動きの癖を覚えた勾配降下

素朴な勾配降下は、勾配の大きさがパラメータによってまちまちだと、うまく進みません。
Adam は、パラメータごとに「これまでの勾配の平均（\( m \)）」と「二乗の平均（\( v \)）」を覚えておき、
\( m / \sqrt{v} \) で更新します。大きく揺れるパラメータは慎重に、静かなパラメータは大胆に動かします。

パラメータは「名前付きの数」なので、Adam の状態も「名前付きの数」が 2 つあるだけです。

```scala mdoc
import nomatrix.nn.Params
import nomatrix.train.{Adam, AdamState}

val start = Params(Map("x" -> 5.0, "y" -> -3.0))

// loss = x^2 + (y - 1)^2 を最小化する。答えは x = 0, y = 1
val (end, _) = (1 to 300).foldLeft((start, AdamState.initial)) { case ((p, s), _) =>
  val pv = p.lift
  val loss = pv("x") * pv("x") + (pv("y") - 1.0) * (pv("y") - 1.0)
  Adam.step(p, pv.gradients(Value.gradients(loss)), s, lr = 0.1)
}
end.values.map((k, v) => k -> f"$v%.3f")
```

`Params` も `AdamState` も不変です。`step` は新しい `Params` と新しい状態を返し、次のステップに渡します。

## 学習ループ

すべてをつなぎます。

1. コーパスから窓を切り出す
2. パラメータを `Value` に持ち上げ、モデルを組み立て、損失を計算
3. 逆伝播して名前付きの勾配を得る
4. Adam で新しいパラメータを作る
5. 1 に戻る

```scala
val (params, _, log) = (1 to steps).foldLeft(start) { case ((p, state, log), step) =>
  val window = sampleWindow(corpus, cfg.context, rng)
  val (loss, grads) = lossAndGradients(cfg, p, window)
  val (next, nextState) = Adam.step(p, grads, state, lr)
  (next, nextState, log :+ TrainStep(step, loss))
}
```

`foldLeft` がループで、状態はすべて引数として渡されます。どこにも `var` がありません。

## 小さく試す

4 種類のトークンが `0 1 2 3 0 1 2 3 ...` と繰り返すだけのコーパスで、40 ステップ学習してみます。

```scala mdoc
import nomatrix.model.{Config, Transformer}

val cfg = Config(vocabSize = 4, dModel = 4, heads = 2, layers = 1, context = 4, hidden = 8)
val toy = Vector.tabulate(64)(i => i % 4)
val rng = new Random(0)
val (_, log) = Trainer.train(cfg, Transformer.init(cfg, rng), toy, steps = 40, lr = 0.05, rng = rng)

log.take(3).map(s => f"step ${s.step}: ${s.loss}%.3f")
log.takeRight(3).map(s => f"step ${s.step}: ${s.loss}%.3f")
```

4 種類なら当てずっぽうの損失は \( \log 4 = 1.386 \)。
規則が完全に分かれば 0 に近づきます。40 ステップでもはっきり下がっています。

## 実装を読む

```scala
--8<-- "src/main/scala/nomatrix/train/Loss.scala"
```

```scala
--8<-- "src/main/scala/nomatrix/train/Adam.scala"
```

```scala
--8<-- "src/main/scala/nomatrix/train/Trainer.scala"
```

!!! tip "この章のまとめ"
    - 損失 = 正解の確率の -log。勾配は「確率 − 正解」
    - 勾配の逆向きに少し動くのが学習。Adam は動きの癖を覚えて調整する
    - パラメータも Adam の状態も「名前付きの数」。ループは `foldLeft`

次は、学習したモデルに文章を書かせます。
