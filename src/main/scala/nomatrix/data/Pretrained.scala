package nomatrix.data

import nomatrix.nn.Params
import scala.io.Source
import scala.util.Using

/** 同梱コーパスで学習済みのパラメータ。チュートリアルで「学習後」の様子を見るために使う。 */
object Pretrained:

  private def load(resource: String): Params =
    Using.resource(Source.fromResource(resource)(using scala.io.Codec.UTF8))(s => Params.parse(s.mkString))

  /** 本線: 微分を使わない「ゆらぎ学習」で学習したもの。 */
  lazy val params: Params = load("pretrained.txt")

  /** 付録: 逆伝播（微分）で 2000 ステップ学習したもの。 */
  lazy val backprop: Params = load("pretrained-backprop.txt")
