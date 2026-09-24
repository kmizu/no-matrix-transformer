# 第10章 損失：どれだけ外したか

モデルは「次の文字のスコア」を出せるようになりました。
でも中身はランダムです。正しい答えに近づけるには、まず「いまの答えがどれだけ外れているか」を
ひとつの数で測る必要があります。それが**損失**です。

## 損失: 正解の確率の -log

位置 \( i \) のロジットを softmax で確率にし、**正解の文字の確率**を見ます。
正解の確率が高いほど良い。損失は「正解の確率の対数にマイナスをつけたもの」です。

\[
\text{loss} = -\log p(\text{正解})
\]

正解の確率が 1 なら損失 0。0.5 なら 0.69。0.01 なら 4.6。これを**交差エントロピー**と呼びます。

```scala mdoc
import nomatrix.train.Loss
import nomatrix.vec.Vec

val logits = Vector(1.0, 2.0, 3.0)
Vec.softmax(logits)                    // 各文字の確率
Loss.crossEntropy(logits, 2)           // 正解が 2 番なら、-log(0.665)
Loss.crossEntropy(logits, 0)           // 正解が 0 番なら、-log(0.090)
```

実装では softmax を経由せず、\( \log \sum_j e^{x_j} - x_{\text{正解}} \) と計算しています。
数学的には同じで、こちらの方が桁あふれに強く、計算も少なくて済みます。

## 当てずっぽうの損失

語彙が \( V \) 種類あって、全部を等確率で予測すると、正解の確率は \( 1/V \)。損失は \( \log V \) です。
同梱コーパスの語彙は 64 文字なので、\( \log 64 \approx 4.16 \)。
これが「何も学んでいない」ときの基準です。学習が進むと、ここから下がっていきます。

```scala mdoc
math.log(64.0)
```

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

## モデルの損失を測る

同梱コーパスの先頭の窓で、ランダムなモデルの損失を測ってみます。

```scala mdoc
import nomatrix.data.Corpus
import nomatrix.model.Transformer

val tok = Corpus.tokenizer
val cfg = Corpus.defaultConfig(tok.vocabSize)
val first = Corpus.ids.take(cfg.context + 1)
tok.decode(first)

val random = Transformer.init(cfg, new Random(0))
Trainer.loss(cfg, random, first)
```

当てずっぽうの 4.16 と、ほぼ同じです。次の章で、この数を小さくしていきます。

## 実装を読む

```scala
--8<-- "src/main/scala/nomatrix/train/Loss.scala"
```

!!! tip "この章のまとめ"
    - 損失 = 正解の確率の -log。小さいほど良い
    - 当てずっぽうの損失は log(語彙数)。同梱コーパスでは 4.16
    - 窓ひとつで context 個の予測問題を解き、平均を取る

次は、損失を小さくする方法。**微分は使いません。**
