package com.htmlism.magicite.sevensegment

import com.htmlism.magicite.*

/**
  * A causal, or source-recovery, experiment: start with a canonical digit, damage one segment, and ask what caused the
  * observed display. The cause is the source digit before the damage.
  *
  * This remains distinct from `NearestCanonicalOracle`. A source-recovery label describes the glyph before damage; it
  * may be ambiguous even when a nearest-glyph recognizer would return a deterministic observed digit.
  */
object CorruptionRecovery:
  /** One observed one-bit corruption and every canonical digit that could have produced it */
  final case class CorruptedDisplay(segments: SegmentState, potentialCanonicalDigits: Vector[Int]):
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
    def trainingExample: SevenSegment.LabeledDisplay =
      require(isUnambiguous, s"cannot train on ambiguous corruption with candidates $potentialCanonicalDigits")

      SevenSegment.LabeledDisplay(potentialCanonicalDigits(0), segments)

  /** A reproducible split of unique, unambiguous corrupted inputs into training and held-out evaluation sets */
  final case class Partition(training: Vector[CorruptedDisplay], evaluation: Vector[CorruptedDisplay]):
    require(
      (training ++ evaluation).forall(_.isUnambiguous),
      "training and evaluation corruptions must be unambiguous"
    )

    /** The labelled noisy examples that may be appended to canonical training rows */
    def trainingExamples: Vector[SevenSegment.LabeledDisplay] =
      training.map:
        _.trainingExample

  /** One held-out prediction together with its acceptable source-digit candidates */
  final case class Prediction(corruption: CorruptedDisplay, predictedDigit: Int):
    /** Whether the prediction is one of the corruption's potential canonical source digits */
    def isPotentialCanonical: Boolean =
      corruption.containsPotentialCanonical(predictedDigit)

  /** Membership-based evaluation for corruptions, including patterns that remain intrinsically ambiguous */
  final case class Evaluation(predictions: Vector[Prediction]):
    /** Fraction of predictions contained in their pattern's potential canonical-digit list */
    def potentialCanonicalAccuracy: Double =
      require(predictions.nonEmpty, "cannot calculate accuracy for an empty evaluation set")

      predictions.count(_.isPotentialCanonical).toDouble / predictions.size

  /** Every distinct one-bit corruption, grouped by observed pattern rather than source row */
  val corruptedDisplays: Vector[CorruptedDisplay] =
    val candidatesBySegments =
      SevenSegment
        .canonicalDigits
        .flatMap: canonical =>
          SevenSegment
            .Segment
            .values
            .indices
            .map: i =>
              canonical
                .segments
                .updated(SevenSegment.Segment.values(i), 1 - canonical.segments.toVector(i)) -> canonical.digit
        .foldLeft(Map.empty[SegmentState, Vector[Int]]):
          case (grouped, (segments, sourceDigit)) =>
            grouped.updated(segments, grouped.getOrElse(segments, Vector.empty) :+ sourceDigit)

    candidatesBySegments
      .toVector
      .map: (segments, sourceDigits) =>
        CorruptedDisplay(segments, sourceDigits.distinct.sorted)
      .sortBy(_.segments.toVector.mkString)

  /** Corruptions that exactly equal a canonical glyph and are excluded from noisy training and evaluation */
  val canonicalShaped: Vector[CorruptedDisplay] =
    corruptedDisplays
      .filter: corruption =>
        SevenSegment.canonicalDigits.exists(_.segments == corruption.segments)

  /** Corruptions that are not themselves canonical glyphs */
  val nonCanonical: Vector[CorruptedDisplay] =
    corruptedDisplays
      .filterNot: corruption =>
        SevenSegment.canonicalDigits.exists(_.segments == corruption.segments)

  /** Corruptions with one possible source label, suitable for ordinary supervised training and exact scoring */
  val unambiguous: Vector[CorruptedDisplay] =
    nonCanonical
      .filter: corruption =>
        corruption.isUnambiguous

  /** Corruptions with two or more source labels, retained for separate ambiguity reporting */
  val ambiguous: Vector[CorruptedDisplay] =
    nonCanonical
      .filterNot: corruption =>
        corruption.isUnambiguous

  /**
    * Randomly partitions unique unambiguous corruption patterns.
    *
    * Grouping happens before this split, so the same observed seven-bit input cannot leak across its boundary.
    */
  def partition(trainingFraction: Double): Rng[Partition] =
    require(trainingFraction > 0.0 && trainingFraction < 1.0, "training fraction must be between zero and one")

    val trainingSize =
      (unambiguous.size * trainingFraction).toInt

    require(trainingSize > 0 && trainingSize < unambiguous.size, "split must leave both partitions non-empty")

    Rng
      .shuffle(unambiguous.toList)
      .map: shuffled =>
        Partition(shuffled.take(trainingSize).toVector, shuffled.drop(trainingSize).toVector)

  /** Trains on every canonical glyph plus the unambiguous corruption rows in one partition */
  def train[A: RealScalar as scalar](
      initialNetwork: SevenSegment.Model[A],
      partition: Partition,
      epochCount: Int,
      learningRate: A
  ): SevenSegment.TrainingResult[A] =
    SevenSegment.train(
      initialNetwork,
      SevenSegment.canonicalDigits ++ partition.trainingExamples,
      epochCount,
      learningRate
    )

  /** Evaluates source recovery by candidate membership rather than nearest-glyph recognition */
  def evaluate[A: RealScalar as scalar](
      network: SevenSegment.Model[A],
      corruptions: Vector[CorruptedDisplay]
  ): Evaluation =
    val predictions =
      corruptions.map: corruption =>
        Prediction(corruption, SevenSegment.predictSegments(network, corruption.segments))

    Evaluation(predictions)
