package com.htmlism.magicite.sevensegment

import scala.util.Random

/**
  * Explores optimization settings for causal source recovery from held-out one-bit corruptions.
  *
  * Run with `sbt 'seven-segment/Test/runMain com.htmlism.magicite.sevensegment.CorruptionRecoveryProbe'`.
  *
  * Each run partitions unique unambiguous corruptions, trains on canonical digits plus that partition's training rows,
  * and scores source-recovery membership accuracy on its held-out rows. The report shows whether a configuration is
  * merely lucky (`perfectRuns` is low) or robust across initializations and partitions.
  *
  * This is a hyperparameter probe, not an acceptance test. Do not choose a final configuration from this report and
  * then present the same held-out rows as an unbiased test result; reserve a final untouched test partition first. It
  * varies epochs and learning rate, but not architecture: hidden width remains the model's fixed sixteen neurons.
  */
object CorruptionRecoveryProbe:
  def main(args: Array[String]): Unit =
    val epochCounts =
      Vector(10, 50, 100, 250, 500, 1_000, 2_000)

    val learningRates =
      Vector(0.03, 0.1, 0.3)

    val modelSeeds =
      0L to 2L

    val partitionSeeds =
      0L to 2L

    println("epochs rate meanHitRate worstHitRate perfectRuns")

    for
      epochCount   <- epochCounts
      learningRate <- learningRates
    do
      val accuracies =
        for
          modelSeed     <- modelSeeds
          partitionSeed <- partitionSeeds
        yield heldOutAccuracy(epochCount, learningRate, modelSeed, partitionSeed)

      val meanHitRate =
        accuracies.sum / accuracies.size

      val worstHitRate =
        accuracies.foldLeft(1.0)(Math.min)

      val perfectRuns =
        accuracies.count(_ == 1.0)

      println(
        f"$epochCount%6d $learningRate%4.2f $meanHitRate%11.3f $worstHitRate%12.3f $perfectRuns%11d/${accuracies.size}"
      )

  private def heldOutAccuracy(
      epochCount: Int,
      learningRate: Double,
      modelSeed: Long,
      partitionSeed: Long
  ): Double =
    val partition =
      CorruptionRecovery
        .partition(trainingFraction = 0.8)
        .runA(Random(partitionSeed))
        .value

    val trained =
      CorruptionRecovery.train(
        initialNetwork = SevenSegment.initialize[Double](modelSeed),
        partition      = partition,
        epochCount     = epochCount,
        learningRate   = learningRate
      )

    CorruptionRecovery.evaluate(trained.network, partition.evaluation).potentialCanonicalAccuracy
