package com.htmlism.magicite.sevensegment

import scala.util.Random

import weaver.*

import com.htmlism.magicite.*
import com.htmlism.magicite.Approx.approximatelyEqual

object SevenSegmentSuite extends FunSuite:
  test("encodes the canonical digits in digit order"):
    val digits =
      SevenSegment
        .canonicalDigits
        .map:
          _.digit

    val segments =
      SevenSegment
        .canonicalDigits
        .map:
          _.segments

    expect.all(
      digits == (0 to 9).toVector,
      segments == Vector(
        Vector(1, 1, 1, 1, 1, 1, 0),
        Vector(0, 1, 1, 0, 0, 0, 0),
        Vector(1, 1, 0, 1, 1, 0, 1),
        Vector(1, 1, 1, 1, 0, 0, 1),
        Vector(0, 1, 1, 0, 0, 1, 1),
        Vector(1, 0, 1, 1, 0, 1, 1),
        Vector(1, 0, 1, 1, 1, 1, 1),
        Vector(1, 1, 1, 0, 0, 0, 0),
        Vector(1, 1, 1, 1, 1, 1, 1),
        Vector(1, 1, 1, 1, 0, 1, 1)
      )
    )

  test("converts a binary row into a typed numeric input vector"):
    val zero =
      SevenSegment.encodeInputs[Double](SevenSegment.canonicalDigits(0))

    expect(zero.values.toVector == Vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0))

  test("encodes a digit target as one hot"):
    val target =
      SevenSegment.encodeTarget[Double](3)

    expect(target.values.toVector == Vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

  test("groups one-bit corruptions before separating canonical-shaped and ambiguous patterns"):
    expect.all(
      SevenSegment.corruptedDisplays.size == 51,
      SevenSegment.canonicalShapedCorruptions.size == 8,
      SevenSegment.nonCanonicalCorruptions.size == 43,
      SevenSegment.unambiguousCorruptions.size == 31,
      SevenSegment.ambiguousCorruptions.size == 12,
      SevenSegment
        .canonicalShapedCorruptions
        .forall: corruption =>
          SevenSegment.canonicalDigits.exists(_.segments == corruption.segments),
      SevenSegment.unambiguousCorruptions.forall(_.isUnambiguous),
      SevenSegment.ambiguousCorruptions.forall(corruption => !corruption.isUnambiguous)
    )

  test("partitions unique unambiguous patterns without leakage"):
    val partition =
      SevenSegment.partitionUnambiguousCorruptions(trainingFraction = 0.8).runA(Random(0)).value

    val trainingPatterns =
      partition
        .training
        .map:
          _.segments
        .toSet

    val evaluationPatterns =
      partition
        .evaluation
        .map:
          _.segments
        .toSet

    val allUnambiguousPatterns =
      SevenSegment
        .unambiguousCorruptions
        .map:
          _.segments

    val trainingLabels =
      partition
        .trainingExamples
        .map:
          _.digit

    val partitionLabels =
      partition
        .training
        .map:
          _.potentialCanonicalDigits(0)

    expect.all(
      partition.training.size == 24,
      partition.evaluation.size == 7,
      trainingPatterns.intersect(evaluationPatterns).isEmpty,
      trainingPatterns ++ evaluationPatterns == allUnambiguousPatterns.toSet,
      trainingLabels == partitionLabels
    )

  test("scores corruption predictions by potential-canonical membership"):
    val evaluation =
      SevenSegment.CorruptionEvaluation(
        Vector(
          SevenSegment.CorruptionPrediction(
            SevenSegment.CorruptedDisplay(Vector(1, 0, 1, 1, 1, 1, 0), Vector(0, 6)),
            predictedDigit = 6
          ),
          SevenSegment.CorruptionPrediction(
            SevenSegment.CorruptedDisplay(Vector(1, 1, 0, 1, 0, 0, 1), Vector(2)),
            predictedDigit = 7
          )
        )
      )

    expect(approximatelyEqual(evaluation.potentialCanonicalAccuracy, 0.5, tolerance = 1e-12))

  /*
   * There are 31 unambiguous corruptions. This deterministic 80/20 split puts 24 in training and leaves 7 held out.
   *
   * `potentialCanonicalAccuracy` scores a top-1 prediction: the most probable digit must be the corruption's sole
   * possible canonical source. The `> 0.5` assertion therefore requires at least 4 of those 7 predictions, which is
   * an intentionally modest "materially above random 10-class guessing" bar for this first generalization slice.
   *
   * It does not assert that every held-out row is correct, nor that the source digit merely has probability above
   * 0.1. A future stronger robustness milestone should require every held-out unambiguous corruption to predict its
   * sole source digit, ideally after checking that result across several model and partition seeds.
   */
  test("training on canonical and partitioned corruptions generalizes to held-out corruptions"):
    val partition =
      SevenSegment
        .partitionUnambiguousCorruptions(trainingFraction = 0.8)
        .runA(Random(0))
        .value

    val trained =
      SevenSegment.trainCanonicalAndCorruptions(
        initialNetwork = SevenSegment.initialize[Double](seed = 0),
        partition      = partition,
        epochCount     = 2_000,
        learningRate   = 0.1
      )

    val evaluation =
      SevenSegment.evaluate(trained.network, partition.evaluation)

    println(
      s"Held-out corruption candidate-membership accuracy: ${evaluation.potentialCanonicalAccuracy} " +
        s"(${evaluation.predictions.count(_.isPotentialCanonical)}/${evaluation.predictions.size})"
    )

    expect.all(
      evaluation.predictions.size == partition.evaluation.size,
      evaluation.potentialCanonicalAccuracy > 0.5
    )

  test("a seeded classifier learns every canonical digit"):
    val trained =
      SevenSegment.train(
        initialNetwork = SevenSegment.initialize[Double](seed = 0),
        epochCount     = 2_000,
        learningRate   = 0.1
      )

    val predictions =
      SevenSegment
        .canonicalDigits
        .map:
          SevenSegment.predict(trained.network, _)

    val probabilities =
      SevenSegment
        .canonicalDigits
        .map:
          SevenSegment.predictProbabilities(trained.network, _)

    val expectedDigits =
      SevenSegment
        .canonicalDigits
        .map:
          _.digit

    expect.all(
      predictions == expectedDigits,
      probabilities.forall(probability => approximatelyEqual(probability.values.sum, 1.0, tolerance = 1e-12))
    )
