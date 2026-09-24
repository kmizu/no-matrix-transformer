# 第11章 学習：試して、良ければ採用

損失を測れるようになりました。あとは、損失が小さくなるようにパラメータ（名前付きの数）を変えるだけです。

普通の解説では、ここで「損失をパラメータで微分して勾配を求め、その逆向きに動かす」と続きます。
この本では微分を使いません。もっと素朴にやります。

## 山登り: ひとつ動かして、良ければ残す

いちばん単純な方法はこうです。

1. パラメータをひとつ選ぶ
2. 少しだけ動かしてみる
3. 損失を測り直す。下がっていたら採用、上がっていたら元に戻す
4. 1 に戻る

第3章で見た「名前で動かす」を、損失を見ながら繰り返すだけです。

```scala mdoc
import nomatrix.data.Corpus
import nomatrix.model.Transformer
import nomatrix.train.Trainer
import scala.util.Random

val tok = Corpus.tokenizer
val cfg = Corpus.defaultConfig(tok.vocabSize)
val window = Corpus.ids.take(cfg.context + 1)
val start = Transformer.init(cfg, new Random(0))

def tryOne(params: nomatrix.nn.Params, name: String, delta: Double): (Double, Double) =
  val before = Trainer.loss(cfg, params, window)
  val after = Trainer.loss(cfg, params.updated(name, params(name) + delta), window)
  (before, after)

tryOne(start, "head.n3.b", +0.5)
tryOne(start, "head.n3.b", -0.5)
```

出力ヘッドの 3 番目のバイアスを +0.5 すると損失が上がり、-0.5 すると下がる（あるいはその逆）。
どちらが良いかは、試せば分かります。これが学習の本質です。

ただし、この方法には問題があります。パラメータは約 7,000 個。1 個ずつ試すと、1 周に 7,000 回の順伝播が要ります。

## 全部いっぺんにゆらす

そこで、**全パラメータに一斉にランダムなゆらぎを足して**試します。

1. 7,000 個の数それぞれに、小さな乱数（平均 0、幅 \( \sigma \)）を足したものを作る
2. 同じ乱数を引いたものも作る（「+ 側」と「− 側」）
3. 両方の損失を測る
4. + 側が良ければゆらぎの向きへ、− 側が良ければ逆向きへ、少し動く

これを何組かのゆらぎで繰り返し、平均した向きに動きます。
1 組あたりに要るのは順伝播 2 回だけ。7,000 個のパラメータの「どっちに動かすと良いか」を、
まとめて、荒っぽく、でも確実に推定しています。

\[
\text{向き} = \frac{1}{n}\sum_{k=1}^{n} \frac{L(\theta + \sigma \epsilon_k) - L(\theta - \sigma \epsilon_k)}{2\sigma}\, \epsilon_k
\]

\( \epsilon_k \) がゆらぎ、\( L \) が損失、\( \theta \) がいまのパラメータです。
式にすると仰々しいですが、コードは「足して、引いて、測って、比べる」だけです。

```scala
val noises = Array.fill(s.pairs)(Array.fill(n)(rng.nextGaussian()))
val scores = noises.map { noise =>
  val plus  = Array.tabulate(n)(i => params(i) + s.sigma * noise(i))
  val minus = Array.tabulate(n)(i => params(i) - s.sigma * noise(i))
  (lossOf(plus), lossOf(minus))
}
```

この方法は**進化戦略**（Evolution Strategies）と呼ばれ、実際に使われています。
2023 年の MeZO という手法は、この考え方で 130 億パラメータの言語モデルを順伝播だけで追加学習しました。
おもちゃではありません。

## 歩幅を整える: Adam

「向き」が分かったら、その向きに少し動きます。どれだけ動くかが**学習率**です。

ただ、ゆらぎで推定した向きはノイズが多く、パラメータによって大きさもまちまちです。
そこで、パラメータごとに「これまでの向きの平均」と「向きの二乗の平均」を覚えておき、
大きく揺れているパラメータは慎重に、静かなパラメータは大胆に動かします。これが Adam です。

Adam は普通「勾配」に対して使いますが、中身は「向きの数列を平滑化する」だけなので、
ゆらぎで推定した向きにもそのまま使えます。この本ではそうしています。

## 小さく試す

4 種類のトークンが `0 1 2 3 0 1 2 3 ...` と繰り返すだけのコーパスで、300 ステップ学習してみます。
このコードは、サイトを生成するたびに実際に実行されています。

```scala mdoc
import nomatrix.model.Config
import nomatrix.train.Evolution

val toyCfg = Config(vocabSize = 4, dModel = 4, heads = 2, layers = 1, context = 4, hidden = 8)
val toy = Vector.tabulate(64)(i => i % 4)
val rng = new Random(0)
val toyStart = Transformer.init(toyCfg, rng)
val (trained, log) = Trainer.train(toyCfg, toyStart, toy, steps = 300,
  Evolution.Settings(pairs = 16, sigma = 0.05, lr = 0.02), windows = 4, rng)

log.filter(_.step % 50 == 0).map(s => f"step ${s.step}%3d  loss ${s.loss}%.3f")
Trainer.loss(toyCfg, toyStart, Vector(0, 1, 2, 3, 0))
Trainer.loss(toyCfg, trained, Vector(0, 1, 2, 3, 0))
```

4 種類なら当てずっぽうの損失は \( \log 4 = 1.386 \)。
規則が完全に分かれば 0 に近づきます。微分を一度も使わずに、はっきり下がっています。

## 学習ループ

すべてをつなぎます。

1. コーパスから窓をいくつか切り出す
2. ゆらぎを何組か作り、+ 側と − 側の損失を測る（窓は全部のゆらぎで共通にする）
3. 向きを推定し、Adam で歩幅を整えて動く
4. 1 に戻る

```scala
val (params, _, log) = (1 to steps).foldLeft(start) { case ((p, state, log), step) =>
  val batch = Vector.fill(windows)(sampleWindow(corpus, cfg.context, rng))
  val lossOf = (candidate: Array[Double]) => batch.map(fast.loss(candidate, _)).sum / batch.length
  val (next, nextState, mean) = Evolution.step(p, lossOf, settings, state, rng)
  (next, nextState, log :+ TrainStep(step, mean))
}
```

`foldLeft` がループで、状態はすべて引数として渡されます。どこにも `var` がありません。

!!! note "速い順伝播"
    1 ステップに順伝播を 100 回以上呼ぶので、学習では `FastTransformer` という配列版の順伝播を使っています。
    計算は第9章の `Transformer` とまったく同じで、パラメータを名前ではなく配列の位置で引くだけです。
    両者が同じロジットを出すことはテストで確かめています。

## 実装を読む

```scala
--8<-- "src/main/scala/nomatrix/train/Evolution.scala"
```

```scala
--8<-- "src/main/scala/nomatrix/train/Trainer.scala"
```

!!! tip "この章のまとめ"
    - 学習 = 少し動かして試し、良かった方へ寄る。それだけ
    - 全パラメータを一斉にゆらして、+ 側と − 側を比べると、向きがまとめて分かる
    - Adam は歩幅の調整役。微分は要らない

次は、学習したモデルに文章を書かせます。
