package nomatrix.nn

import nomatrix.vec.Vec
import scala.util.Random

/** FeedForward = 広げて（Dense）、折り曲げて（ReLU）、戻す（Dense）。トークンごとに独立に働く。 */
final case class FeedForward(up: Dense, down: Dense):
  def apply(x: Vec): Vec = down(Vec.relu(up(x)))

object FeedForward:

  def init(prefix: String, dModel: Int, hidden: Int, rng: Random): Params =
    Dense.init(s"$prefix.up", dModel, hidden, rng) ++ Dense.init(s"$prefix.down", hidden, dModel, rng)

  def load(p: ParamValues, prefix: String, dModel: Int, hidden: Int): FeedForward =
    FeedForward(Dense.load(p, s"$prefix.up", dModel, hidden), Dense.load(p, s"$prefix.down", hidden, dModel))
