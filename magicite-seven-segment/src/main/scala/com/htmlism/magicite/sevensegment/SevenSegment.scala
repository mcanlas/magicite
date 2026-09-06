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

  /** One labelled canonical digit input for classification */
  final case class CanonicalDigit(digit: Int, segments: Vector[Int]):
    require(0 <= digit && digit <= 9, s"digit must be between 0 and 9, but was $digit")
    require(
      segments.length == Segment.values.length,
      s"a display must have ${Segment.values.length} segments, but had ${segments.length}"
    )
    require(segments.forall(bit => bit == 0 || bit == 1), s"segments must be binary, but were $segments")

  /**
    * The ten authoritative digit inputs in `[top, upper-right, lower-right, bottom, lower-left, upper-left, middle]`
    * order
    */
  val canonicalDigits: Vector[CanonicalDigit] =
    Vector(
      CanonicalDigit(0, Vector(1, 1, 1, 1, 1, 1, 0)),
      CanonicalDigit(1, Vector(0, 1, 1, 0, 0, 0, 0)),
      CanonicalDigit(2, Vector(1, 1, 0, 1, 1, 0, 1)),
      CanonicalDigit(3, Vector(1, 1, 1, 1, 0, 0, 1)),
      CanonicalDigit(4, Vector(0, 1, 1, 0, 0, 1, 1)),
      CanonicalDigit(5, Vector(1, 0, 1, 1, 0, 1, 1)),
      CanonicalDigit(6, Vector(1, 0, 1, 1, 1, 1, 1)),
      CanonicalDigit(7, Vector(1, 1, 1, 0, 0, 0, 0)),
      CanonicalDigit(8, Vector(1, 1, 1, 1, 1, 1, 1)),
      CanonicalDigit(9, Vector(1, 1, 1, 1, 0, 1, 1))
    )

  /** The clean-only classifier shape: seven binary inputs, sixteen hidden neurons, and ten output logits */
  type Model[A] = Network.WithHidden[A, Segments, Dimension.D16, DigitClasses]

  /** The result of training the canonical-digit classifier for whole epochs */
  final case class TrainingResult[A](network: Model[A], epochLosses: Vector[A])

  /** Converts a canonical binary segment row into a numeric network input vector */
  def encodeInputs[A: RealScalar as scalar](row: CanonicalDigit): Vec[A, Segments] =
    Vec[A, Segments](scalar.tabulate(Segment.values.length): i =>
      scalar.fromDouble(row.segments(i).toDouble))

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

  /** Performs one categorical-cross-entropy update for a canonical digit */
  def trainRow[A: RealScalar as scalar](
      network: Model[A],
      row: CanonicalDigit,
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

  /** Trains the classifier with fixed-order stochastic-gradient epochs over the canonical digits */
  def train[A: RealScalar as scalar](
      initialNetwork: Model[A],
      epochCount: Int,
      learningRate: A
  ): TrainingResult[A] =
    require(epochCount > 0, s"epoch count must be positive, but was $epochCount")
    require(scalar.isPositive(learningRate), "learning rate must be positive")

    val (network, epochLosses) =
      (0 until epochCount).foldLeft(initialNetwork -> Vector.empty[A]):
        case ((currentNetwork, losses), _) =>
          val (updatedNetwork, totalLoss) =
            canonicalDigits.foldLeft(currentNetwork -> scalar.zero):
              case ((network, loss), row) =>
                val (updatedNetwork, rowLoss) =
                  trainRow(network, row, learningRate)

                updatedNetwork -> (loss + rowLoss)

          updatedNetwork -> (losses :+ totalLoss / scalar.fromDouble(canonicalDigits.size.toDouble))

    TrainingResult(network, epochLosses)

  /** Converts the network's ten logits for a digit into class probabilities */
  def predictProbabilities[A: RealScalar](network: Model[A], row: CanonicalDigit): Vec[A, DigitClasses] =
    Softmax.probabilities(network.forward(encodeInputs[A](row)).prediction)

  /** Returns the index of the largest predicted class probability */
  def predict[A: RealScalar as scalar](network: Model[A], row: CanonicalDigit): Int =
    val probabilities =
      predictProbabilities(network, row).values

    probabilities
      .indices
      .drop(1)
      .foldLeft(0): (largestIndex, index) =>
        if scalar.isPositive(probabilities(index) - probabilities(largestIndex)) then index else largestIndex

  private val DigitClassesDimension = 10
