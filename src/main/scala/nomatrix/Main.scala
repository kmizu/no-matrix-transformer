package nomatrix

import nomatrix.data.Corpus
import nomatrix.gen.Generator
import nomatrix.model.Transformer
import nomatrix.nn.Params
import nomatrix.train.{Evolution, Trainer}
import java.nio.file.{Files, Path}
import scala.util.Random

/** CLI: `train` で学習してパラメータを保存、`generate` で続きを書かせる。 */
object Main:

  final case class Options(
      command: String,
      steps: Int = 4000,
      lr: Double = 0.003,
      pairs: Int = 64,
      sigma: Double = 0.05,
      windows: Int = 8,
      backprop: Boolean = false,
      paramsPath: String = "params.txt",
      prompt: String = "あさ ",
      count: Int = 80,
      temperature: Double = 0.8,
      seed: Long = 0L
  )

  def parse(args: List[String]): Either[String, Options] =
    def loop(rest: List[String], opts: Options): Either[String, Options] = rest match
      case Nil                          => Right(opts)
      case "--steps" :: v :: tail       => v.toIntOption.toRight(s"--steps は整数: $v").flatMap(n => loop(tail, opts.copy(steps = n)))
      case "--lr" :: v :: tail          => v.toDoubleOption.toRight(s"--lr は数: $v").flatMap(d => loop(tail, opts.copy(lr = d)))
      case "--pairs" :: v :: tail       => v.toIntOption.toRight(s"--pairs は整数: $v").flatMap(n => loop(tail, opts.copy(pairs = n)))
      case "--sigma" :: v :: tail       => v.toDoubleOption.toRight(s"--sigma は数: $v").flatMap(d => loop(tail, opts.copy(sigma = d)))
      case "--windows" :: v :: tail     => v.toIntOption.toRight(s"--windows は整数: $v").flatMap(n => loop(tail, opts.copy(windows = n)))
      case "--backprop" :: tail         => loop(tail, opts.copy(backprop = true))
      case "--params" :: v :: tail      => loop(tail, opts.copy(paramsPath = v))
      case "--prompt" :: v :: tail      => loop(tail, opts.copy(prompt = v))
      case "--count" :: v :: tail       => v.toIntOption.toRight(s"--count は整数: $v").flatMap(n => loop(tail, opts.copy(count = n)))
      case "--temperature" :: v :: tail => v.toDoubleOption.toRight(s"--temperature は数: $v").flatMap(d => loop(tail, opts.copy(temperature = d)))
      case "--seed" :: v :: tail        => v.toLongOption.toRight(s"--seed は整数: $v").flatMap(n => loop(tail, opts.copy(seed = n)))
      case other :: _                   => Left(s"不明な引数: $other")
    args match
      case ("train" | "generate") :: tail => loop(tail, Options(args.head))
      case _ =>
        Left("使い方: train|generate [--steps N] [--lr X] [--pairs N] [--sigma X] [--windows N] [--backprop] [--params FILE] [--prompt TEXT] [--count N] [--temperature T] [--seed N]")

  def run(opts: Options, out: String => Unit): Unit =
    val tokenizer = Corpus.tokenizer
    val cfg = Corpus.defaultConfig(tokenizer.vocabSize)
    val rng = new Random(opts.seed)
    opts.command match
      case "train" =>
        val method = if opts.backprop then "逆伝播" else "ゆらぎ学習（微分なし）"
        out(s"語彙 ${tokenizer.vocabSize} 文字、パラメータ ${Transformer.parameterCount(cfg)} 個、${opts.steps} ステップ、$method")
        val every = math.max(1, opts.steps / 20)
        val log = (step: Int, loss: Double) => if step % every == 0 then out(f"step $step%5d  loss $loss%.4f")
        val params =
          if opts.backprop then
            nomatrix.backprop.Trainer.train(cfg, Transformer.init(cfg, rng), Corpus.ids, opts.steps, opts.lr, rng, s => log(s.step, s.loss))._1
          else
            Trainer.train(cfg, Transformer.init(cfg, rng), Corpus.ids, opts.steps,
              Evolution.Settings(opts.pairs, opts.sigma, opts.lr), opts.windows, rng, s => log(s.step, s.loss))._1
        Files.writeString(Path.of(opts.paramsPath), params.toText)
        out(s"保存: ${opts.paramsPath}")
      case _ =>
        val params = Params.parse(Files.readString(Path.of(opts.paramsPath)))
        val ids = Generator.generate(cfg, params, tokenizer.encode(opts.prompt), opts.count, opts.temperature, rng)
        out(tokenizer.decode(ids))

  def main(args: Array[String]): Unit =
    parse(args.toList) match
      case Left(msg)   => System.err.println(msg); sys.exit(1)
      case Right(opts) => run(opts, println)
