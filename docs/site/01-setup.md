# 第1章 準備

## 必要なもの

| もの | バージョン | 用途 |
|---|---|---|
| JDK | 17 以上 | Scala を動かす |
| sbt | 1.11 系 | ビルド・テスト |
| Scala | 3.3（sbt が自動で取得） | 本文のコード |

Scala 3 を選んだ理由は、`for` 式やタプルの分解、`extension` など、
「数と関数」だけで書くのに向いた道具が揃っているからです。

## リポジトリを取ってくる

```bash
git clone https://github.com/kmizu/no-matrix-transformer.git
cd no-matrix-transformer
sbt test
```

`sbt test` が通れば準備完了です。60 個ほどのテストが数十秒で終わります。

## プロジェクトの形

```text
src/main/scala/nomatrix/
├── vec/Vec.scala            数の並びの操作（第2章）
├── nn/
│   ├── Params.scala         名前付きの数の集まり（第3章）
│   ├── Dense.scala          ニューロン（第3章）
│   ├── Embedding.scala      埋め込み（第4-5章）
│   ├── AttentionHead.scala  注意（第6章）
│   ├── MultiHead.scala      複数ヘッド（第7章）
│   ├── FeedForward.scala    FeedForward（第8章）
│   ├── LayerNorm.scala      LayerNorm（第8章）
│   └── Block.scala          ブロック（第8章）
├── model/
│   ├── Transformer.scala    全体（第9章）
│   └── FastTransformer.scala 同じ計算の配列版（第11章）
├── data/                    トークナイザとコーパス（第4章）
├── train/
│   ├── Loss.scala           損失（第10章）
│   ├── Evolution.scala      ゆらぎ学習（第11章）
│   └── Trainer.scala        学習ループ（第11章）
├── gen/Generator.scala      生成（第12章）
├── Main.scala               CLI（第13章）
└── backprop/                付録: 微分できる数で書き直したもの
```

本線（`backprop/` 以外）はひとつのファイルが 30〜120 行、全部足しても 700 行ほどです。

## このサイトのコードについて

本文中のコードブロックのうち、出力が `// ` で始まる行として付いているものは、
サイトを生成するときに [mdoc](https://scalameta.org/mdoc/) が実際にコンパイル・実行したものです。
たとえばこれは、いま本当に計算された結果です。

```scala mdoc
val x = 1 + 2
```

コードが変われば出力も変わります。「説明と実物がずれている」ことが起きない仕組みです。

## 「行列を使わない」「微分を使わない」の定義

この本での約束を、もう少し正確に決めておきます。

- **使わない**: 2 次元配列（`Array[Array[Double]]` など）、行列型、行列積・転置などの行列演算
- **使う**: ひとつの数 `Double`、数の並び `Vector[Double]`
- **本線では使わない**: 微分。学習は「少し動かして試す」だけで行う。微分を使う速い学習法は付録に隔離する

「トークンごとにベクトルがある」ので、`Vector[Vector[Double]]` という形のデータは登場します。
でもそれは「文の各位置に、数の並びがひとつずつある」というだけで、行列としては扱いません。
行列積を書きたくなる場面は、すべて「ニューロンが並んでいる」「似ている度合いを測る」という言葉で書きます。

この定義が守られているかは `NoMatrixTest` が検査しています。

```scala
--8<-- "src/test/scala/nomatrix/NoMatrixTest.scala"
```

次は、すべての土台になる「数の並び」から始めます。
