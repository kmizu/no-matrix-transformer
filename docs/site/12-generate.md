# 第12章 文章を生成する

学習したモデルは、「ここまでの文字列を見て、次の文字のスコア」を出します。
文章を書かせるには、それを 1 文字ずつ繰り返すだけです。

## 1 文字ずつ足していく

1. プロンプト（最初の文字列）をトークン列にする
2. 末尾 `context` 文字をモデルに入れ、**最後の位置**のロジットを取る
3. ロジットから 1 文字選び、列の末尾に足す
4. 2 に戻る

```scala
(1 to count).foldLeft(prompt) { (ids, _) =>
  val window = ids.takeRight(cfg.context)
  val next = sample(model.logits(window).last, temperature, rng)
  ids :+ next
}
```

これも `foldLeft`。生成した列を引数として次に渡します。

## 温度: 冒険するか、堅実にいくか

ロジットから 1 文字選ぶとき、いちばんスコアの高い文字を毎回選ぶと、同じ文が繰り返されがちです。
softmax の確率で**抽選**すると、多様になります。

その中間を決めるのが**温度**です。ロジットを温度で割ってから softmax にかけます。

- 温度 0: 常に最大のスコアを選ぶ（決定的）
- 温度 1: softmax の確率そのままで抽選
- 温度 0.5: 差が強調され、堅実寄り
- 温度 2: 差が薄まり、冒険寄り

```scala mdoc
import nomatrix.gen.Generator
import nomatrix.vec.Vec
import scala.util.Random

val logits = Vector(1.0, 3.0, 2.0)

def histogram(temperature: Double): Map[Int, Int] =
  val rng = new Random(0)
  (1 to 1000).map(_ => Generator.sample(logits, temperature, rng)).groupBy(identity).map((k, v) => k -> v.size)

histogram(0.0)
histogram(0.5)
histogram(1.0)
histogram(2.0)
```

温度 0 では 1 番だけ。温度が上がるほど、0 番や 2 番も選ばれるようになります。

## 動かしてみる

まだ学習していないモデルでも、生成の仕組み自体は動きます。

```scala mdoc
import nomatrix.model.{Config, Transformer}
import nomatrix.data.Tokenizer

val tok = Tokenizer.fromText("あいうえお")
val cfg = Config(vocabSize = tok.vocabSize, dModel = 4, heads = 2, layers = 1, context = 4, hidden = 8)
val params = Transformer.init(cfg, new Random(1))

val ids = Generator.generate(cfg, params, tok.encode("あ"), count = 10, temperature = 1.0, rng = new Random(2))
tok.decode(ids)
```

ランダムなモデルなので、ランダムな文字列です。学習したあとの出力は次章で見ます。

## 実装を読む

```scala
--8<-- "src/main/scala/nomatrix/gen/Generator.scala"
```

!!! note "KV キャッシュについて"
    本物の推論エンジンは、毎回全文をモデルに通し直すのではなく、
    過去のトークンの key と value を覚えておいて再利用します（KV キャッシュ）。
    この本では分かりやすさを優先して、毎回通し直しています。
    キャッシュも「位置ごとの数の並びを覚えておく」だけなので、行列は要りません。

!!! tip "この章のまとめ"
    - 生成 = 「最後の位置のロジットから 1 文字選んで足す」の繰り返し
    - 温度で「堅実」と「冒険」を調整する
    - `foldLeft` で、生成した列を次に渡す

次は、実際に同梱コーパスで学習して、出力の変化を見ます。
