# 第9章 ブロックを積む：Transformer

部品が揃いました。組み立てます。

## 全体の流れ

1. 文字の埋め込み + 位置の埋め込み（第4-5章）
2. ブロック × `layers`（第8章）
3. 最後に LayerNorm で整える
4. 語彙の数だけニューロンを並べた `Dense` で、「次の文字はどれか」のスコア（**ロジット**）を出す

```scala
def logits(ids: Vector[Int]): Tokens =
  val hidden = blocks.foldLeft(embed(ids))((xs, block) => block(xs))
  hidden.map(x => head(finalNorm(x)))
```

`foldLeft` でブロックを順に通す。それだけです。

```mermaid
flowchart TB
  ids["トークン列 [12, 8, 3]"] --> emb["埋め込み + 位置"]
  emb --> b0["Block 0"] --> b1["Block 1"] --> ln["LayerNorm"]
  ln --> head["Dense head<br/>(語彙数個のニューロン)"]
  head --> logits["各位置のロジット<br/>(語彙数個の数)"]
```

## 出力は「各位置ごと」に出る

入力が 3 トークンなら、出力も 3 本のベクトルです。
位置 \( i \) のロジットは「位置 \( i \) までを見たとき、位置 \( i+1 \) は何か」の予測です。
学習では全位置を同時に使い（第10-11章）、生成では最後の位置だけを使います（第12章）。

## Config: モデルの大きさ

```scala
final case class Config(vocabSize: Int, dModel: Int, heads: Int, layers: Int, context: Int, hidden: Int)
```

この本の既定値です。

| 名前 | 値 | 意味 |
|---|---|---|
| `vocabSize` | コーパスによる（同梱コーパスでは 64） | 文字の種類 |
| `dModel` | 16 | 各トークンのベクトルの長さ |
| `heads` | 2 | 注意ヘッドの数（各 8 次元） |
| `layers` | 2 | ブロックの段数 |
| `context` | 16 | 一度に見る文字数 |
| `hidden` | 32 | FeedForward の中間の長さ |

## 動かしてみる

```scala mdoc
import nomatrix.model.*
import scala.util.Random

val cfg = Config(vocabSize = 7, dModel = 4, heads = 2, layers = 2, context = 5, hidden = 8)
val params = Transformer.init(cfg, new Random(5))
params.size

val model = Transformer.load(cfg, params)
val out = model.logits(Vector(1, 2, 3))
out.length
out.map(_.map(d => f"$d%.3f"))
```

3 トークン入れて、各位置に 7 個（語彙数）のスコアが出ました。まだ学習していないので意味はありません。

## パラメータ数を数える

「約 7,000 個」と言ってきた根拠を出します。既定の設定で数えてみます。

```scala mdoc
import nomatrix.data.Corpus

val real = Corpus.defaultConfig(Corpus.tokenizer.vocabSize)
Transformer.parameterCount(real)
```

内訳を名前の先頭で集計すると、どこに数が多いか分かります。

```scala mdoc
Transformer.init(real, new Random(0)).names
  .groupBy(_.split('.').head)
  .map((k, v) => k -> v.size)
  .toVector.sortBy(-_._2)
```

ブロック 1 つあたり約 2,200 個。埋め込みと出力ヘッドは語彙数に比例します。

## ひとつの数が、端から端まで効く

パラメータをひとつ動かすと、出力は変わるでしょうか。
いちばん入口にある「トークン 1 の埋め込みの 0 番目」を少し動かして、最後のロジットを見比べます。

```scala mdoc
val before = model.logits(Vector(1, 2))(1)
val moved = params.updated("tok.t1.d0", params("tok.t1.d0") + 0.5)
val after = Transformer.load(cfg, moved).logits(Vector(1, 2))(1)
before.zip(after).map((b, a) => f"${a - b}%+.3f")
```

位置 1 の出力は、注意を通じて位置 0（トークン 1）の情報を受け取っているので、
入口の数ひとつが 2 段のブロックを通り抜けて出口まで届いています。
第11章の学習は、この「動かすと出力が変わる」を使って、損失が下がる方へ数を寄せていきます。

## 実装を読む

```scala
--8<-- "src/main/scala/nomatrix/model/Transformer.scala"
```

50 行足らずです。行列がないぶん、部品の名前がそのまま構造になっています。

!!! tip "この章のまとめ"
    - 埋め込み → ブロックを `foldLeft` → LayerNorm → 語彙ぶんのニューロン
    - 出力は各位置ごとの「次の文字のスコア」
    - パラメータは約 7,000 個、すべて名前付きの数。ひとつ動かせば出口まで効く

次は、このモデルの答えに点数をつける「損失」です。
