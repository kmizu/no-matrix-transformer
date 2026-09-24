# 第4章 トークンと埋め込み

言語モデルは文字列を直接は扱えません。数の並びにする必要があります。
その入口が**トークナイザ**と**埋め込み**です。

## トークン = 文字に番号をふる

このチュートリアルでは、いちばん単純な「1 文字 = 1 トークン」を採用します。
コーパスに出てくる文字を集めて並べ、番号をふるだけです。

```scala mdoc
import nomatrix.data.Tokenizer

val tok = Tokenizer.fromText("ねこが ねる。とりが とぶ。")
tok.chars
tok.vocabSize
tok.encode("ねこ")
tok.decode(tok.encode("とりが ねる"))
```

同梱のコーパスはひらがな中心の短い文章です。

```scala mdoc
import nomatrix.data.Corpus

Corpus.tokenizer.vocabSize
Corpus.ids.length
Corpus.text.linesIterator.take(3).mkString("\n")
```

!!! note "単語やサブワードでなく文字を使う理由"
    語彙が小さければ、出力側のニューロン（第9章）も少なくて済み、CPU で学習できます。
    トークナイザの方式は Transformer 本体とは独立なので、後から差し替えられます。

## 埋め込み = 番号から数の並びを引く

番号のままでは「似ている文字」を表現できません。
そこで、**番号ごとに数の並び（ベクトル）を 1 本用意し**、それを引き当てます。これが埋め込みです。

```scala
final case class Embedding(table: Vector[Vec]):
  def apply(id: Int): Vec = table(id)
```

「表を引く」だけ。行列の掛け算で書く解説もありますが（one-hot ベクトルとの積）、
やっていることは配列の添字アクセスです。

```scala mdoc
import nomatrix.nn.Embedding
import nomatrix.vec.Vec
import scala.util.Random

val params = Embedding.init("tok", count = 4, dim = 3, new Random(0))
params.names.toVector.sorted.take(6)

val emb = Embedding.load(params, "tok", count = 4, dim = 3)
emb(2)
```

`tok.t2.d1` は「2 番のトークンの、1 番目の次元」。
この数もパラメータなので、学習で変わります。**似た使われ方をする文字は、似たベクトルに育っていく**。
それが埋め込みの面白いところです。

## 何を学ぶのか

最初はランダムなベクトルです。学習が進むと、たとえば「が」と「を」（助詞）の
ベクトルが近づき、「。」と「\n」（区切り）が近づく、といったことが起こります。
第13章で実際に学習したあとの埋め込みを、内積で見比べてみます。

## 実装を読む

```scala
--8<-- "src/main/scala/nomatrix/data/Tokenizer.scala"
```

```scala
--8<-- "src/main/scala/nomatrix/nn/Embedding.scala"
```

!!! tip "この章のまとめ"
    - トークナイザ = 文字に番号をふる
    - 埋め込み = 番号ごとに数の並びを持つ表。引くだけ
    - 表の中身はパラメータ。学習で「似た文字が似たベクトルに」なる

次は、トークンの「位置」をどう伝えるかです。
