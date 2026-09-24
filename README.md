# 行列を使わない Transformer 入門

行列（2 次元配列・行列積）も微分も使わずに、Transformer ベースの小さな言語モデルを Scala で組み立てるチュートリアルです。

**サイト:** https://kmizu.github.io/no-matrix-transformer/

サイトに載っているコードは、すべて [mdoc](https://scalameta.org/mdoc/) により実際にコンパイル・実行され、出力が埋め込まれています。

## 何があるか

- `src/main/scala/nomatrix/` — 数の並び `Vec`、ニューロン・注意・FeedForward・LayerNorm・Transformer、微分を使わない学習（ゆらぎ学習）、生成、CLI。`backprop/` は付録（微分できる数 `Value` と逆伝播）
- `src/test/scala/nomatrix/` — munit によるテスト。`NoMatrixTest` がソースに行列型が無いことを検査
- `docs/site/` — チュートリアル本文（mdoc 入り Markdown）
- `mkdocs.yml` — MkDocs Material の設定

## 動かす

```bash
sbt test                                                    # テスト
sbt "runMain nomatrix.Main train --steps 4000"              # 学習（微分なし、params.txt に保存）
sbt "runMain nomatrix.Main train --backprop --steps 2000"   # 付録: 逆伝播で学習
sbt "runMain nomatrix.Main generate --prompt あさ --count 80" # 生成
```

## サイトをローカルで作る

```bash
pip install mkdocs-material
sbt docs/mdoc
mkdocs serve
```

## ライセンス

MIT
