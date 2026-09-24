# 第13章 実際に学習させる

部品はすべて揃い、学習と生成の仕組みもできました。同梱のコーパスで本当に学習させてみます。

## 設定

```scala mdoc
import nomatrix.data.Corpus
import nomatrix.model.Transformer

val tok = Corpus.tokenizer
val cfg = Corpus.defaultConfig(tok.vocabSize)
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

## このページの上で 150 ステップ学習する

以下のコードは、このサイトを生成するたびに mdoc が実際に実行しています。
CPU で 150 ステップ、十数秒です。

```scala mdoc
import nomatrix.train.Trainer

val rng = new Random(42)
val (after150, log) = Trainer.train(cfg, Transformer.init(cfg, rng), Corpus.ids, steps = 150, lr = 0.01, rng = rng)

log.filter(_.step % 25 == 0).map(s => f"step ${s.step}%3d  loss ${s.loss}%.3f")
```

損失が当てずっぽうの \( \log 64 \approx 4.16 \) から下がっているのが分かります。
150 ステップ後の出力はこうです。

```scala mdoc
tok.decode(Generator.generate(cfg, after150, tok.encode("あさ"), count = 40, temperature = 0.8, rng = new Random(1)))
```

まだ文にはなっていませんが、「が」「を」「。」の使われ方や、空白の入り方が
コーパスに似てきているはずです。

## 2000 ステップ学習したモデル

CLI で 2000 ステップ学習したパラメータを `src/main/resources/pretrained.txt` に同梱しています。

```bash
sbt "runMain nomatrix.Main train --steps 2000 --lr 0.01 --seed 7"
```

手元（ノート PC の CPU、1 コア）での学習ログの抜粋です。

--8<-- "docs/site/snippets/train2000.md"

損失は 200 ステップで 2.0、600 ステップで 0.8 前後まで下がり、その後は 0.7〜1.7 の間を揺れます。
窓ごとに難しさが違うので、1 ステップごとの損失はばらつきます。

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
コーパスにない組み合わせも作ります。文字単位で「次の文字」を学んだだけで、
単語の切れ目、助詞、句点の位置を覚えたわけです。

## 埋め込みは何を学んだか

第5章で「似た使われ方をする文字は、似たベクトルに育つ」と書きました。確かめます。
文字の埋め込みを取り出し、内積を長さで割った値（コサイン類似度）で「似ている度合い」を測ります。

```scala mdoc
import nomatrix.nn.Embedding
import nomatrix.vec.Vec

val emb = Embedding.load(trained.lift, "tok", tok.vocabSize, cfg.dModel)

def cosine(a: Char, b: Char): Double =
  val va = Vec.data(emb(tok.encode(a.toString).head))
  val vb = Vec.data(emb(tok.encode(b.toString).head))
  val dot = va.zip(vb).map(_ * _).sum
  dot / (math.sqrt(va.map(x => x * x).sum) * math.sqrt(vb.map(x => x * x).sum))

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
第7章の `AttentionHead` は最終結果しか返さないので、同じ計算をこの場で書き下します。

```scala mdoc
import nomatrix.nn.*

val model = Transformer.load(cfg, trained.lift)
val text = "ねこが ねる。"
val ids = tok.encode(text)
val embedded = ids.zipWithIndex.map((id, pos) => Vec.add(model.tokenEmbedding(id), model.positionEmbedding(pos)))
val block0 = model.blocks(0)
val normed = embedded.map(block0.norm1(_))
val head0 = block0.attention.heads(0)

val qs = normed.map(head0.query(_))
val ks = normed.map(head0.key(_))
val last = ids.length - 1
val scores = (0 to last).toVector.map(j => Vec.dot(qs(last), ks(j)) * (1.0 / math.sqrt(head0.headDim.toDouble)))
val weights = Vec.data(Vec.softmax(scores))

text.zip(weights).map((c, w) => f"'$c' ${w}%.2f")
```

最後の文字「。」が、それより前のどの文字にどれだけ注目しているかの割合です。
合計は 1 になります。どこに重みが寄っているかは、モデルが「次を決めるのに何を見ているか」の手がかりです。

## 自分で試す

```bash
# 学習（params.txt に保存）
sbt "runMain nomatrix.Main train --steps 2000 --lr 0.01"

# 生成
sbt "runMain nomatrix.Main generate --prompt あさ --count 80 --temperature 0.6"
```

`src/main/resources/corpus.txt` を書き換えれば、別の文章で学習できます。
語彙が増えるとパラメータと時間が増えるので、最初は 100 種類以下の文字に抑えるのがおすすめです。

!!! tip "この章のまとめ"
    - 当てずっぽうの損失 \( \log 64 \approx 4.16 \) から、2000 ステップで 1 前後まで下がる
    - 文字単位でも、単語の切れ目・助詞・句点の位置を覚える
    - 埋め込みと注意の重みを覗くと、「何を学んだか」の手がかりが見える

最終章では、ここまで一度も使わなかった「行列」が本当は何だったのかを整理します。
