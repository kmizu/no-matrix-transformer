# 第13章 実際に学習させる

部品はすべて揃い、学習と生成の仕組みもできました。同梱のコーパスで本当に学習させてみます。
**微分は使いません。** 第11章の「ゆらぎを試して、良かった方へ寄る」だけです。

## 設定

```scala mdoc:silent
import nomatrix.data.Corpus
import nomatrix.model.Transformer

val tok = Corpus.tokenizer
val cfg = Corpus.defaultConfig(tok.vocabSize)
```

```scala mdoc
tok.chars.mkString
cfg
Transformer.parameterCount(cfg)
```

当てずっぽう（全文字を等確率で予測）の損失は \( \log(\text{語彙数}) \) です。

```scala mdoc
math.log(tok.vocabSize.toDouble)
```

## 学習前の出力

ランダムなパラメータで「あさ」の続きを書かせると、こうなります。

```scala mdoc
import nomatrix.gen.Generator
import scala.util.Random

val untrained = Transformer.init(cfg, new Random(0))
tok.decode(Generator.generate(cfg, untrained, tok.encode("あさ"), count = 40, temperature = 0.8, rng = new Random(1)))
```

文字がでたらめに並ぶだけです。

## このページの上で 100 ステップ学習する

以下のコードは、このサイトを生成するたびに mdoc が実際に実行しています。
1 ステップに順伝播を 8 窓 × 16 対 × 2 = 256 回呼ぶので、100 ステップで十数秒です。

```scala mdoc
import nomatrix.train.{Evolution, Trainer}

val rng = new Random(42)
val (after100, log) = Trainer.train(cfg, Transformer.init(cfg, rng), Corpus.ids, steps = 100,
  Evolution.Settings(pairs = 16, sigma = 0.02, lr = 0.003), windows = 8, rng)

log.filter(_.step % 20 == 0).map(s => f"step ${s.step}%3d  loss ${s.loss}%.3f")
```

損失が当てずっぽうの \( \log 64 \approx 4.16 \) から下がり始めているのが分かります。
ゆらぎ学習は 1 ステップの情報が少ないので、ここから先は時間がかかります。

## 8000 ステップ学習したモデル

CLI で 8000 ステップ学習したパラメータを `src/main/resources/pretrained.txt` に同梱しています。

```bash
sbt "runMain nomatrix.Main train --steps 8000 --pairs 64 --sigma 0.02 --lr 0.003 --windows 8 --seed 7"
```

手元（ノート PC の CPU、1 コア）での学習ログの抜粋です。損失は学習に使っていない固定の 64 窓で測っています。

--8<-- "docs/site/snippets/evolution.md"

窓ごとに難しさが違うので、1 ステップごとの損失はばらつきます。
それでも、当てずっぽうの 4.16 から、微分を一度も使わずに 1 台前半まで下がりました。

学習済みパラメータで生成してみます。これも mdoc がその場で実行しています。

```scala mdoc
import nomatrix.data.Pretrained

val trained = Pretrained.params

def write(prompt: String, temperature: Double, seed: Int): String =
  tok.decode(Generator.generate(cfg, trained, tok.encode(prompt), count = 60, temperature = temperature, rng = new Random(seed)))

write("あさ", 0.3, 1)
write("あさ", 0.8, 1)
write("よる", 0.5, 2)
write("ねこ", 0.5, 3)
```

「あさ おきて」「ごはんを たべる」「ねこが ねる」のような、コーパスにある語の並びが出てきます。
コーパスにない組み合わせも作ります。文字単位で「次の文字」を、ゆらぎを試すだけで学んだ結果です。

## 埋め込みは何を学んだか

第4章で「似た使われ方をする文字は、似たベクトルに育つ」と書きました。確かめます。
文字の埋め込みを取り出し、内積を長さで割った値（コサイン類似度）で「似ている度合い」を測ります。

```scala mdoc
import nomatrix.nn.Embedding
import nomatrix.vec.Vec

val emb = Embedding.load(trained, "tok", tok.vocabSize, cfg.dModel)

def cosine(a: Char, b: Char): Double =
  val va = emb(tok.encode(a.toString).head)
  val vb = emb(tok.encode(b.toString).head)
  Vec.dot(va, vb) / (math.sqrt(Vec.dot(va, va)) * math.sqrt(Vec.dot(vb, vb)))

def nearest(c: Char): Vector[(Char, String)] =
  tok.chars.filter(_ != c).map(o => o -> f"${cosine(c, o)}%.2f").sortBy(-_._2.toDouble).take(3)

nearest('が')
nearest('。')
nearest('る')
```

助詞「が」の近くに他の助詞が来たり、句点「。」の近くに改行が来たりしていれば、
モデルは「文字の役割」を数の並びとして覚えたことになります。
（乱数の種によって結果は変わります。ここに出ているのは、いま実際に計算された値です。）

## 注意は何を見ているか

最後に、学習済みモデルの注意の重みを覗いてみます。
第6章の `AttentionHead.weights` は、「トークン i が自分以前の各トークンにどれだけ注目するか」を返します。

```scala mdoc
val model = Transformer.load(cfg, trained)
val text = "ねこが ねる。"
val ids = tok.encode(text)
val block0 = model.blocks(0)
val normed = model.embed(ids).map(block0.norm1(_))
val head0 = block0.attention.heads(0)
val last = ids.length - 1

text.zip(head0.weights(normed, last)).map((c, w) => f"'$c' ${w}%.2f")
```

最後の文字「。」が、それより前のどの文字にどれだけ注目しているかの割合です。
合計は 1 になります。どこに重みが寄っているかは、モデルが「次を決めるのに何を見ているか」の手がかりです。

## 自分で試す

```bash
# 学習（params.txt に保存）。ノート PC で 1 時間ほど
sbt "runMain nomatrix.Main train --steps 8000"

# 生成
sbt "runMain nomatrix.Main generate --prompt あさ --count 80 --temperature 0.6"
```

`src/main/resources/corpus.txt` を書き換えれば、別の文章で学習できます。
語彙が増えるとパラメータと時間が増えるので、最初は 100 種類以下の文字に抑えるのがおすすめです。

!!! tip "この章のまとめ"
    - 当てずっぽうの損失 \( \log 64 \approx 4.16 \) から、微分なしの 8000 ステップで 1 台前半まで下がる
    - 文字単位でも、単語の切れ目・助詞・句点の位置を覚える
    - 埋め込みと注意の重みを覗くと、「何を学んだか」の手がかりが見える

最終章では、ここまで一度も使わなかった「行列」と「微分」が本当は何だったのかを整理します。
