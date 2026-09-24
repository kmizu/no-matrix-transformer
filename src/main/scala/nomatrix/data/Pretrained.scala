package nomatrix.data

import nomatrix.nn.Params
import scala.io.Source
import scala.util.Using

/** 同梱コーパスで 2000 ステップ学習したパラメータ。チュートリアルで「学習後」の様子を見るために使う。 */
object Pretrained:

  lazy val params: Params =
    Using.resource(Source.fromResource("pretrained.txt")(using scala.io.Codec.UTF8))(s => Params.parse(s.mkString))
