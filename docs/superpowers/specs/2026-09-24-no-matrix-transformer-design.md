# 行列を使わない Transformer 入門 — 設計書

日付: 2026-09-24

## 目的

「Transformer は行列演算の塊」という先入観を外し、**行列（2次元配列・行列積）を一切使わずに**
Transformer ベースの小さな言語モデル（SLM）を Scala で組み立てる過程を、GitHub Pages 上の
チュートリアルとして公開する。

読者: プログラミング経験はあるが線形代数に苦手意識がある人。Scala 未経験でも読める粒度にする。

成功条件:

1. リポジトリ内の Scala コードに行列型・行列積が存在しない（`Array[Array[Double]]` 等を使わない）。
2. チュートリアルに載せたコードは mdoc により実際にコンパイル・実行され、出力が埋め込まれる。
3. 小さなコーパスで学習し、損失が下がり、学習後に生成テキストが変化することをページ上で示す。
4. `sbt test` が通り、GitHub Actions でサイトが GitHub Pages に配信される。

## 方針: 「すべてはスカラー」

- 数は `Double`、微分可能な数は `Value`（micrograd 方式のスカラー自動微分ノード）。
- ベクトルは「数の並び」 `Vector[Value]`。行列は登場しない。
- 「線形層」は「ニューロンの集まり」。ニューロンは入力の重み付き和＋バイアス。
- 注意（attention）は「各トークンが他のトークンとどれだけ似ているかを内積で測り、その割合で混ぜる」。
- 行列は「たくさんの重み付き和をまとめて書く記法／高速化」であることを最終章で明かす。

## サイトジェネレータ

**mdoc + MkDocs Material** を採用。

- mdoc: `docs/` 配下の Markdown 内 `scala mdoc` ブロックをコンパイル・実行し `target/mdoc` に出力。
- MkDocs Material: `target/mdoc` を `docs_dir` として静的サイトを生成。日本語検索、MathJax、Mermaid。
- GitHub Actions: `sbt docs/mdoc` → `mkdocs build` → `actions/deploy-pages`。

不採用: Docusaurus（Node 依存、コード検証なし）、Jekyll（機能が弱い）、Laika（日本語検索なし）、Hugo（コード実行なし）。

## アーキテクチャ（Scala 3、sbt、munit）

パッケージ `nomatrix`:

| モジュール | 役割 |
|---|---|
| `autograd.Value` | スカラー自動微分。`data` と `(入力, 局所微分)` の列を持つ不変ノード。`Value.gradients(root)` で逆伝播。 |
| `vec.Vec` | `Vector[Value]` に対する add / scale / dot / softmax / layerNorm / relu。 |
| `nn.Params` | パラメータ = 名前付きの数 `Map[String, Double]`。順伝播開始時に `Value` の葉へ持ち上げる。 |
| `nn.Dense` | ニューロンの集まり。`init` と `apply`。 |
| `nn.Embedding` | トークン ID → ベクトルの引き当て表。位置埋め込みも同じ仕組み。 |
| `nn.Attention` | 因果的単一ヘッド注意。 |
| `nn.MultiHead` | 複数ヘッドの結果を連結し Dense で混ぜる。 |
| `nn.FeedForward` | Dense → ReLU → Dense。 |
| `nn.LayerNorm` | 平均 0 分散 1 に整えて gain / bias。 |
| `nn.Block` | LayerNorm → 注意 → 残差、LayerNorm → FF → 残差。 |
| `model.Transformer` | 埋め込み → Block × N → LayerNorm → 語彙へのニューロン（ロジット）。 |
| `data.Tokenizer` | 文字単位トークナイザ。 |
| `train.Loss` | 交差エントロピー（logsumexp で安定化）。 |
| `train.Adam` | 名前付きの数に対する不変な Adam 更新。 |
| `train.Trainer` | 学習ループ。 |
| `gen.Generator` | 温度付きサンプリング生成。 |
| `Main` | CLI: `train` / `generate`。 |

モデル既定値: d_model=16, heads=2, layers=2, context=16, ff=32。文字単位、ひらがな中心の自作コーパス。

## テスト

- munit。各モジュールに単体テスト。`Value` は数値微分との一致で検証。
- 統合テスト: 小コーパスで数十ステップ学習し損失が単調に近く下がること。
- mdoc のビルド自体がドキュメントのコードの結合テストになる。

## サイト構成（章）

1. はじめに — なぜ行列を使わないのか
2. 準備 — Scala と sbt
3. 微分できる数 `Value`
4. ベクトルは数の並び
5. ニューロンと Dense
6. トークンと埋め込み
7. 位置を知る
8. 注意 — 似ているものを混ぜる
9. 複数の視点 — Multi-Head
10. FeedForward・残差・LayerNorm
11. ブロックを積む — Transformer
12. 損失と学習 — Adam
13. 文章を生成する
14. 実際に学習させてみる
15. おわりに — 行列はどこにいたのか
