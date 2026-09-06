package com.htmlism.magicite.sevensegment

import scala.util.Random

import com.htmlism.magicite.*

/** The canonical seven-segment digit inputs used by the classification experiment */
object SevenSegment:
  /** The seven input features, in their fixed vector order */
  enum Segment:
    case Top, UpperRight, LowerRight, Bottom, LowerLeft, UpperLeft, Middle

  /** Phantom dimension for a seven-segment display's input vector */
  sealed trait Segments

  /** Phantom dimension for the ten digit-class logits */
  sealed trait DigitClasses

  given Dimension[Segments]     = Dimension(Segment.values.length)
  given Dimension[DigitClasses] = Dimension(10)

  /** One labelled seven-segment input, whether canonical or an unambiguous corruption */
  final case class LabeledDisplay(digit: Int, segments: Vector[Int]):
    require(0 <= digit && digit <= 9, s"digit must be between 0 and 9, but was $digit")
    requireSegments(segments)

  /**
    * The ten authoritative digit inputs in `[top, upper-right, lower-right, bottom, lower-left, upper-left, middle]`
    * order
    */
  val canonicalDigits: Vector[LabeledDisplay] =
    Vector(
      LabeledDisplay(0, Vector(1, 1, 1, 1, 1, 1, 0)),
      LabeledDisplay(1, Vector(0, 1, 1, 0, 0, 0, 0)),
      LabeledDisplay(2, Vector(1, 1, 0, 1, 1, 0, 1)),
      LabeledDisplay(3, Vector(1, 1, 1, 1, 0, 0, 1)),
      LabeledDisplay(4, Vector(0, 1, 1, 0, 0, 1, 1)),
      LabeledDisplay(5, Vector(1, 0, 1, 1, 0, 1, 1)),
      LabeledDisplay(6, Vector(1, 0, 1, 1, 1, 1, 1)),
      LabeledDisplay(7, Vector(1, 1, 1, 0, 0, 0, 0)),
      LabeledDisplay(8, Vector(1, 1, 1, 1, 1, 1, 1)),
      LabeledDisplay(9, Vector(1, 1, 1, 1, 0, 1, 1))
    )

  /** The clean-only classifier shape: seven binary inputs, sixteen hidden neurons, and ten output logits */
  type Model[A] = Network.WithHidden[A, Segments, Dimension.D16, DigitClasses]

  /** The result of training the canonical-digit classifier for whole epochs */
  final case class TrainingResult[A](network: Model[A], epochLosses: Vector[A])

  /** Converts a labelled binary segment row into a numeric network input vector */
  def encodeInputs[A: RealScalar as scalar](row: LabeledDisplay): Vec[A, Segments] =
    encodeSegments(row.segments)

  /** Encodes a digit class as a ten-element one-hot target vector */
  def encodeTarget[A: RealScalar as scalar](digit: Int): Vec[A, DigitClasses] =
    require(0 <= digit && digit < DigitClassesDimension, s"digit must be between 0 and 9, but was $digit")

    Vec[A, DigitClasses](scalar.tabulate(DigitClassesDimension): i =>
      if i == digit then scalar.one else scalar.zero)

  /** Initializes a seeded `7 → 16 → 10` network whose output layer emits logits */
  def initialize[A: RealScalar](seed: Long): Model[A] =
    Network
      .withHidden[A, Segments, Dimension.D16, DigitClasses](
        initialization   = Initialization.Xavier,
        hiddenLayerCount = 1,
        hiddenActivation = Activation.Tanh,
        outputActivation = Activation.Identity
      )
      .runA(Random(seed))
      .value

  /** Performs one categorical-cross-entropy update for one labelled display input */
  def trainRow[A: RealScalar as scalar](
      network: Model[A],
      row: LabeledDisplay,
      learningRate: A
  ): (Model[A], A) =
    val forwardPass =
      network.forward(encodeInputs[A](row))

    val target =
      encodeTarget[A](row.digit)

    val probabilities =
      Softmax.probabilities(forwardPass.prediction)

    val loss =
      CategoricalCrossEntropy.value(target, probabilities)

    val outputGradient =
      CategoricalCrossEntropy.softmaxPreActivationDerivative(target, probabilities)

    network.updated(network.backwardFromOutputPreActivation(forwardPass, outputGradient), learningRate) -> loss

  /**
    * Trains the classifier with fixed-order, one-example gradient-descent epochs over supplied labelled inputs.
    *
    * This generic learning loop deliberately has no opinion about how examples were collected or labelled.
    */
  def train[A: RealScalar as scalar](
      initialNetwork: Model[A],
      trainingExamples: Vector[LabeledDisplay],
      epochCount: Int,
      learningRate: A
  ): TrainingResult[A] =
    require(trainingExamples.nonEmpty, "training examples must be non-empty")
    require(epochCount > 0, s"epoch count must be positive, but was $epochCount")
    require(scalar.isPositive(learningRate), "learning rate must be positive")

    val (network, epochLosses) =
      (0 until epochCount).foldLeft(initialNetwork -> Vector.empty[A]):
        case ((currentNetwork, losses), _) =>
          val (updatedNetwork, totalLoss) =
            trainingExamples.foldLeft(currentNetwork -> scalar.zero):
              case ((network, loss), row) =>
                val (updatedNetwork, rowLoss) =
                  trainRow(network, row, learningRate)

                updatedNetwork -> (loss + rowLoss)

          updatedNetwork -> (losses :+ totalLoss / scalar.fromDouble(trainingExamples.size.toDouble))

    TrainingResult(network, epochLosses)

  /** Trains only on the ten canonical glyphs, retaining the clean-classifier experiment as a convenient baseline */
  def train[A: RealScalar as scalar](
      initialNetwork: Model[A],
      epochCount: Int,
      learningRate: A
  ): TrainingResult[A] =
    train(initialNetwork, canonicalDigits, epochCount, learningRate)

  /** Converts the network's ten logits for a labelled display into class probabilities */
  def predictProbabilities[A: RealScalar](network: Model[A], row: LabeledDisplay): Vec[A, DigitClasses] =
    Softmax.probabilities(network.forward(encodeInputs[A](row)).prediction)

  /** Returns the index of the largest predicted class probability */
  def predict[A: RealScalar as scalar](network: Model[A], row: LabeledDisplay): Int =
    predictSegments(network, row.segments)

  private[sevensegment] def predictSegments[A: RealScalar as scalar](network: Model[A], segments: Vector[Int]): Int =
    val probabilities =
      Softmax.probabilities(network.forward(encodeSegments[A](segments)).prediction).values

    probabilities
      .indices
      .drop(1)
      .foldLeft(0): (largestIndex, i) =>
        if scalar.isPositive(probabilities(i) - probabilities(largestIndex)) then i else largestIndex

  private def encodeSegments[A: RealScalar as scalar](segments: Vector[Int]): Vec[A, Segments] =
    requireSegments(segments)

    Vec[A, Segments](scalar.tabulate(Segment.values.length): i =>
      scalar.fromDouble(segments(i).toDouble))

  private val DigitClassesDimension = 10

  private[sevensegment] def requireSegments(segments: Vector[Int]): Unit =
    require(
      segments.length == Segment.values.length,
      s"a display must have ${Segment.values.length} segments, but had ${segments.length}"
    )
    require(segments.forall(bit => bit == 0 || bit == 1), s"segments must be binary, but were $segments")
