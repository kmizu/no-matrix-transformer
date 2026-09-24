# 第2章 ベクトルは数の並び

「ベクトル」と聞くと矢印や座標を思い浮かべるかもしれませんが、
このチュートリアルではもっと素朴に扱います。**ベクトルとは、数が並んだもの**です。

```scala
type Vec = Vector[Double]
```

これだけです。`Vector` は Scala の不変なリスト。中身はただの数（`Double`）です。
特別な型も、ライブラリも要りません。

## 使う操作は 6 つ

Transformer を作るのに必要なベクトル操作は、驚くほど少ないです。

| 操作 | 意味 | 使う場所 |
|---|---|---|
| `add(a, b)` | 要素ごとに足す | 残差接続、埋め込みの合成 |
| `mul(a, b)` | 要素ごとに掛ける | LayerNorm のゲイン |
| `scale(v, k)` | 全部を k 倍 | 注意の重み付け |
| `dot(a, b)` | かけて足す（内積） | ニューロン、似ている度合い |
| `softmax(v)` | 合計 1 の割合にする | 注意、次のトークンの確率 |
| `layerNorm(v)` | 平均 0・分散 1 に整える | ブロックの入口 |

```scala mdoc
import nomatrix.vec.Vec

val a = Vector(1.0, 2.0, 3.0)
val b = Vector(4.0, 5.0, 6.0)

Vec.add(a, b)
Vec.dot(a, b)
```

## 内積 = 「似ている度合い」

`dot` は、対応する要素をかけて全部足すだけです。

\[
\text{dot}(a, b) = \sum_i a_i \, b_i
\]

なぜこれが「似ている度合い」なのか。2 つのベクトルが同じ方向を向いていれば、
各要素の符号が揃うので、積がすべて正になり、合計が大きくなります。
逆向きなら合計は負になります。関係がなければゼロの近くをうろつきます。

```scala mdoc
val same      = Vector(1.0, 1.0, -1.0)
val opposite  = Vector(-1.0, -1.0, 1.0)
val unrelated = Vector(1.0, -1.0, 0.0)

Vec.dot(same, same)
Vec.dot(same, opposite)
Vec.dot(same, unrelated)
```

第6章の「注意」は、この内積で「どのトークンに注目するか」を決めます。

## softmax = 「割合にする」

数の並びを「全部足すと 1 になる正の割合」に変えます。
大きい数ほど大きな割合になり、差は指数的に強調されます。

\[
\text{softmax}(v)_i = \frac{e^{v_i}}{\sum_j e^{v_j}}
\]

```scala mdoc
Vec.softmax(Vector(1.0, 2.0, 3.0))
Vec.softmax(Vector(1.0, 2.0, 10.0))
```

実装では、`exp` する前に最大値を引いています。
`exp(1000)` は `Double` では無限大になりますが、`exp(1000 - 1000) = 1` なら安全です。
softmax の値は「差」だけで決まるので、全部から同じ数を引いても結果は変わりません。

## layerNorm = 「整える」

ベクトルを平均 0・分散 1 になるように引き伸ばしたり縮めたりします。
層を重ねると数のスケールが暴れやすいので、各ブロックの入口で整えます。

```scala mdoc
val n = Vec.layerNorm(Vector(10.0, 20.0, 30.0, 40.0))
n.sum / n.size                                     // 平均 ≈ 0
n.map(x => x * x).sum / n.size                     // 分散 ≈ 1
```

## 実装を読む

```scala
--8<-- "src/main/scala/nomatrix/vec/Vec.scala"
```

`zip` して `map` する。`sum` する。それだけです。
「行列」という言葉が出てこないだけでなく、出てくる余地がありません。

!!! tip "この章のまとめ"
    - ベクトル = `Vector[Double]`、数の並び
    - 内積 = かけて足す = 似ている度合い
    - softmax = 割合にする、layerNorm = 整える

次は、この操作だけで「ニューロン」を作ります。
