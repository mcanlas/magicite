package com.htmlism.magicite.sevensegment

import scala.util.Random

import weaver.*

import com.htmlism.magicite.Approx.approximatelyEqual

/** Tests for the causal corruption-and-source-recovery experiment */
object CorruptionRecoverySuite extends FunSuite:
  test("groups one-bit corruptions before separating canonical-shaped and ambiguous patterns"):
    expect.all(
      CorruptionRecovery.corruptedDisplays.size == 51,
      CorruptionRecovery.canonicalShaped.size == 8,
      CorruptionRecovery.nonCanonical.size == 43,
      CorruptionRecovery.unambiguous.size == 31,
      CorruptionRecovery.ambiguous.size == 12,
      CorruptionRecovery
        .canonicalShaped
        .forall: corruption =>
          SevenSegment.canonicalDigits.exists(_.segments == corruption.segments),
      CorruptionRecovery.unambiguous.forall(_.isUnambiguous),
      CorruptionRecovery.ambiguous.forall(corruption => !corruption.isUnambiguous)
    )

  test("partitions unique unambiguous patterns without leakage"):
    val partition =
      CorruptionRecovery.partition(trainingFraction = 0.8).runA(Random(0)).value

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
      CorruptionRecovery
        .unambiguous
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

  test("scores source-recovery predictions by potential-canonical membership"):
    val evaluation =
      CorruptionRecovery.Evaluation(
        Vector(
          CorruptionRecovery.Prediction(
            CorruptionRecovery.CorruptedDisplay(SegmentState(1, 0, 1, 1, 1, 1, 0), Vector(0, 6)),
            predictedDigit = 6
          ),
          CorruptionRecovery.Prediction(
            CorruptionRecovery.CorruptedDisplay(SegmentState(1, 1, 0, 1, 0, 0, 1), Vector(2)),
            predictedDigit = 7
          )
        )
      )

    expect(approximatelyEqual(evaluation.potentialCanonicalAccuracy, 0.5, tolerance = 1e-12))

  /*
   * There are 31 unambiguous corruptions. This deterministic 80/20 split puts 24 in training and leaves 7 held out.
   *
   * `potentialCanonicalAccuracy` scores a top-1 source-recovery prediction: the most probable digit must be the
   * corruption's sole possible canonical source. The `> 0.5` assertion therefore requires at least 4 of those 7
   * predictions, an intentionally modest "materially above random 10-class guessing" bar for this first slice.
   *
   * It does not assert every held-out row is correct, nor that the source merely has probability above 0.1. A future
   * stronger milestone should require every held-out unambiguous corruption to predict its sole source digit across
   * several model and partition seeds.
   */
  test("training on canonical and partitioned corruptions generalizes to held-out source recovery"):
    val partition =
      CorruptionRecovery.partition(trainingFraction = 0.8).runA(Random(0)).value

    val trained =
      CorruptionRecovery.train(
        initialNetwork = SevenSegment.initialize[Double](seed = 0),
        partition      = partition,
        epochCount     = 2_000,
        learningRate   = 0.1
      )

    val evaluation =
      CorruptionRecovery.evaluate(trained.network, partition.evaluation)

    println(
      s"Held-out source-recovery candidate-membership accuracy: ${evaluation.potentialCanonicalAccuracy} " +
        s"(${evaluation.predictions.count(_.isPotentialCanonical)}/${evaluation.predictions.size})"
    )

    expect.all(
      evaluation.predictions.size == partition.evaluation.size,
      evaluation.potentialCanonicalAccuracy > 0.5
    )
