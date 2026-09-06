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
    * One observed one-bit corruption and every canonical digit that could have produced it.
    *
    * A corruption with one potential digit can be used as a supervised training row. More than one potential digit is
    * intrinsically ambiguous: assigning it a single label would train the same seven-bit input toward conflicting
    * answers.
    */
  final case class CorruptedDisplay(segments: Vector[Int], potentialCanonicalDigits: Vector[Int]):
    requireSegments(segments)
    require(potentialCanonicalDigits.nonEmpty, "a corruption must have at least one potential canonical digit")
    require(
      potentialCanonicalDigits == potentialCanonicalDigits.distinct.sorted,
      s"potential canonical digits must be distinct and sorted, but were $potentialCanonicalDigits"
    )

    /** Whether exactly one canonical digit could have produced this observed pattern */
    def isUnambiguous: Boolean =
      potentialCanonicalDigits.size == 1

    /** Whether a prediction is consistent with at least one possible source digit */
    def containsPotentialCanonical(prediction: Int): Boolean =
      potentialCanonicalDigits.contains(prediction)

    /** Converts an unambiguous corruption into a normal supervised input row */
    def trainingExample: LabeledDisplay =
      require(isUnambiguous, s"cannot train on ambiguous corruption with candidates $potentialCanonicalDigits")

      LabeledDisplay(potentialCanonicalDigits(0), segments)

  /** A reproducible split of unique, unambiguous corrupted inputs into training and held-out evaluation sets */
  final case class CorruptionPartition(training: Vector[CorruptedDisplay], evaluation: Vector[CorruptedDisplay]):
    require(
      (training ++ evaluation).forall(_.isUnambiguous),
      "training and evaluation corruptions must be unambiguous"
    )

    /** The labelled noisy examples that may be appended to the canonical training rows */
    def trainingExamples: Vector[LabeledDisplay] =
      training.map:
        _.trainingExample

  /** One held-out prediction together with its acceptable source-digit candidates */
  final case class CorruptionPrediction(corruption: CorruptedDisplay, predictedDigit: Int):
    /** Whether the prediction is one of the corruption's potential canonical source digits */
    def isPotentialCanonical: Boolean =
      corruption.containsPotentialCanonical(predictedDigit)

  /** Membership-based evaluation for corruptions, including patterns that remain intrinsically ambiguous */
  final case class CorruptionEvaluation(predictions: Vector[CorruptionPrediction]):
    /** Fraction of predictions contained in their pattern's potential canonical-digit list */
    def potentialCanonicalAccuracy: Double =
      require(predictions.nonEmpty, "cannot calculate accuracy for an empty evaluation set")

      predictions.count(_.isPotentialCanonical).toDouble / predictions.size

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

  /**
    * Every distinct one-bit corruption, grouped by observed segment pattern rather than source row.
    *
    * Grouping first is essential: different source digits can produce the same damaged pattern, and that pattern must
    * never leak into both the training and held-out evaluation partitions.
    */
  val corruptedDisplays: Vector[CorruptedDisplay] =
    val candidatesBySegments =
      canonicalDigits
        .flatMap: canonical =>
          Segment
            .values
            .indices
            .map: i =>
              canonical.segments.updated(i, 1 - canonical.segments(i)) -> canonical.digit
        .foldLeft(Map.empty[Vector[Int], Vector[Int]]):
          case (grouped, (segments, sourceDigit)) =>
            grouped.updated(segments, grouped.getOrElse(segments, Vector.empty) :+ sourceDigit)

    candidatesBySegments
      .toVector
      .map: (segments, sourceDigits) =>
        CorruptedDisplay(segments, sourceDigits.distinct.sorted)
      .sortBy(_.segments.mkString)

  /** Corrupted patterns that exactly equal a canonical glyph and are excluded from noisy training and evaluation */
  val canonicalShapedCorruptions: Vector[CorruptedDisplay] =
    corruptedDisplays
      .filter: corruption =>
        canonicalDigits.exists(_.segments == corruption.segments)

  /** Corrupted patterns that are not themselves canonical glyphs */
  val nonCanonicalCorruptions: Vector[CorruptedDisplay] =
    corruptedDisplays
      .filterNot: corruption =>
        canonicalDigits.exists(_.segments == corruption.segments)

  /**
    * Noisy patterns with one possible source label; these are safe for ordinary supervised training and exact scoring
    */
  val unambiguousCorruptions: Vector[CorruptedDisplay] =
    nonCanonicalCorruptions
      .filter: corruption =>
        corruption.isUnambiguous

  /** Noisy patterns with two or more source labels; report them separately instead of training contradictory labels */
  val ambiguousCorruptions: Vector[CorruptedDisplay] =
    nonCanonicalCorruptions
      .filterNot: corruption =>
        corruption.isUnambiguous

  /**
    * Randomly partitions unique unambiguous corruption patterns.
    *
    * The result is a stateful `Rng` program so callers can choose a seed and reproduce the split. The training fraction
    * is applied after grouping, preventing duplicate corrupted inputs from crossing the split boundary.
    */
  def partitionUnambiguousCorruptions(trainingFraction: Double): Rng[CorruptionPartition] =
    require(trainingFraction > 0.0 && trainingFraction < 1.0, "training fraction must be between zero and one")

    val trainingSize =
      (unambiguousCorruptions.size * trainingFraction).toInt

    require(
      trainingSize > 0 && trainingSize < unambiguousCorruptions.size,
      "split must leave both partitions non-empty"
    )

    Rng
      .shuffle(unambiguousCorruptions.toList)
      .map: shuffled =>
        CorruptionPartition(shuffled.take(trainingSize).toVector, shuffled.drop(trainingSize).toVector)

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
    * Pass `canonicalDigits ++ partition.trainingExamples` to train on the canonical glyphs together with one
    * partition's unambiguous corruptions.
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

  /**
    * Trains on every canonical glyph plus the training side of an unambiguous corruption partition.
    *
    * Held-out and ambiguous corruptions are deliberately absent: the former measure generalization, while the latter
    * have no single correct supervised label.
    */
  def trainCanonicalAndCorruptions[A: RealScalar as scalar](
      initialNetwork: Model[A],
      partition: CorruptionPartition,
      epochCount: Int,
      learningRate: A
  ): TrainingResult[A] =
    train(initialNetwork, canonicalDigits ++ partition.trainingExamples, epochCount, learningRate)

  /** Converts the network's ten logits for a labelled display into class probabilities */
  def predictProbabilities[A: RealScalar](network: Model[A], row: LabeledDisplay): Vec[A, DigitClasses] =
    Softmax.probabilities(network.forward(encodeInputs[A](row)).prediction)

  /** Returns the index of the largest predicted class probability */
  def predict[A: RealScalar as scalar](network: Model[A], row: LabeledDisplay): Int =
    predictSegments(network, row.segments)

  /** Evaluates corruptions by membership in their potential canonical source-digit lists */
  def evaluate[A: RealScalar as scalar](
      network: Model[A],
      corruptions: Vector[CorruptedDisplay]
  ): CorruptionEvaluation =
    val predictions =
      corruptions.map: c =>
        CorruptionPrediction(c, predictSegments(network, c.segments))

    CorruptionEvaluation(predictions)

  private def predictSegments[A: RealScalar as scalar](network: Model[A], segments: Vector[Int]): Int =
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

  private def requireSegments(segments: Vector[Int]): Unit =
    require(
      segments.length == Segment.values.length,
      s"a display must have ${Segment.values.length} segments, but had ${segments.length}"
    )
    require(segments.forall(bit => bit == 0 || bit == 1), s"segments must be binary, but were $segments")
