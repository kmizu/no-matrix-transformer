package nomatrix.backprop

import nomatrix.model.Config
import nomatrix.nn.Params

import scala.util.Random


/** Transformer 言語モデル。
  *
  *   トークン埋め込み + 位置埋め込み → Block × layers → LayerNorm → 語彙へのニューロン（ロジット）
  */
final case class Transformer(
    cfg: Config,
    tokenEmbedding: Embedding,
    positionEmbedding: Embedding,
    blocks: Vector[Block],
    finalNorm: LayerNorm,
    head: Dense
):

  /** 各位置について「次のトークンは何か」のスコア（ロジット）を返す。 */
  def logits(ids: Vector[Int]): Tokens =
    require(ids.length <= cfg.context, s"長さ ${ids.length} は context=${cfg.context} を超えている")
    require(ids.forall(id => id >= 0 && id < cfg.vocabSize), "語彙の範囲外の id がある")
    val embedded = ids.zipWithIndex.map((id, pos) => GVec.add(tokenEmbedding(id), positionEmbedding(pos)))
    val hidden = blocks.foldLeft(embedded)((xs, block) => block(xs))
    hidden.map(x => head(finalNorm(x)))

object Transformer:

  def init(cfg: Config, rng: Random): Params =
    val blocks = (0 until cfg.layers).map(i => Block.init(s"block$i", cfg.dModel, cfg.heads, cfg.hidden, rng))
    Embedding.init("tok", cfg.vocabSize, cfg.dModel, rng) ++
      Embedding.init("pos", cfg.context, cfg.dModel, rng) ++
      blocks.foldLeft(Params.empty)(_ ++ _) ++
      LayerNorm.init("norm", cfg.dModel) ++
      Dense.init("head", cfg.dModel, cfg.vocabSize, rng)

  def load(cfg: Config, p: ParamValues): Transformer =
    Transformer(
      cfg,
      Embedding.load(p, "tok", cfg.vocabSize, cfg.dModel),
      Embedding.load(p, "pos", cfg.context, cfg.dModel),
      (0 until cfg.layers).toVector.map(i => Block.load(p, s"block$i", cfg.dModel, cfg.heads, cfg.hidden)),
      LayerNorm.load(p, "norm", cfg.dModel),
      Dense.load(p, "head", cfg.dModel, cfg.vocabSize)
    )

  def parameterCount(cfg: Config): Int = init(cfg, new Random(0)).size
