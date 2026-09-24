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
├── autograd/Value.scala     微分できる数（第2章）
├── vec/Vec.scala            数の並びの操作（第3章）
├── nn/
│   ├── Params.scala         名前付きの数の集まり（第4章）
│   ├── Dense.scala          ニューロン（第4章）
│   ├── Embedding.scala      埋め込み（第5-6章）
│   ├── AttentionHead.scala  注意（第7章）
│   ├── MultiHead.scala      複数ヘッド（第8章）
│   ├── FeedForward.scala    FeedForward（第9章）
│   ├── LayerNorm.scala      LayerNorm（第9章）
│   └── Block.scala          ブロック（第9章）
├── model/Transformer.scala  全体（第10章）
├── data/                    トークナイザとコーパス（第5章）
├── train/                   損失・Adam・学習ループ（第11章）
├── gen/Generator.scala      生成（第12章）
└── Main.scala               CLI（第13章）
```

ひとつのファイルは 30〜100 行です。全部足しても 700 行ほどです。

## このサイトのコードについて

本文中のコードブロックのうち、出力が `// ` で始まる行として付いているものは、
サイトを生成するときに [mdoc](https://scalameta.org/mdoc/) が実際にコンパイル・実行したものです。
たとえばこれは、いま本当に計算された結果です。

```scala mdoc
val x = 1 + 2
```

コードが変われば出力も変わります。「説明と実物がずれている」ことが起きない仕組みです。

## 「行列を使わない」の定義

この本での約束を、もう少し正確に決めておきます。

- **使わない**: 2 次元配列（`Array[Array[Double]]` など）、行列型、行列積・転置などの行列演算
- **使う**: ひとつの数 `Double`、微分できる数 `Value`、数の並び `Vector[Value]`

「トークンごとにベクトルがある」ので、`Vector[Vector[Value]]` という形のデータは登場します。
でもそれは「文の各位置に、数の並びがひとつずつある」というだけで、行列としては扱いません。
行列積を書きたくなる場面は、すべて「ニューロンが並んでいる」「似ている度合いを測る」という言葉で書きます。

この定義が守られているかは `NoMatrixTest` が検査しています。

```scala
--8<-- "src/test/scala/nomatrix/NoMatrixTest.scala"
```

次は、すべての土台になる「微分できる数」を作ります。
