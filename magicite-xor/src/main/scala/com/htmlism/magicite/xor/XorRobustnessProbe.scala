package com.htmlism.magicite.xor

/**
  * Samples fixed-size XOR training over seeds, learning rates, and epoch counts
  *
  * This is an exploratory executable, intentionally separate from the deterministic integration test
  */
object XorRobustnessProbe:
  @main def probeXorRobustness(): Unit =
    val seeds =
      0L to 20L

    for
      epochCount   <- Vector(500, 1_000, 2_000)
      learningRate <- Vector(0.1, 0.2, 0.3, 0.5)
    do
      val doubleSuccesses =
        seeds.filter: seed =>
          succeedsDouble(Xor.train(Xor.initialize[Double](seed), epochCount, learningRate))

      val floatSuccesses =
        seeds.filter: seed =>
          succeedsFloat(Xor.train(Xor.initialize[Float](seed), epochCount, learningRate.toFloat))

      println(
        s"epochs=$epochCount rate=$learningRate double=${doubleSuccesses.size}/${seeds.size} float=${floatSuccesses.size}/${seeds.size}"
      )

  private def succeedsDouble(trained: Xor.TrainingResult[Double]): Boolean =
    trained.epochLosses match
      case firstLoss +: remainingLosses =>
        val lastLoss =
          remainingLosses.lastOption.getOrElse(firstLoss)

        val probabilities =
          Xor
            .truthTable
            .map: row =>
              Xor.predictProbability(trained.network, row.left, row.right)

        lastLoss < firstLoss * 0.1 && probabilities(0) < 0.1 && probabilities(1) > 0.9 && probabilities(
          2
        ) > 0.9 && probabilities(3) < 0.1
      case _ => false

  private def succeedsFloat(trained: Xor.TrainingResult[Float]): Boolean =
    trained.epochLosses match
      case firstLoss +: remainingLosses =>
        val lastLoss =
          remainingLosses.lastOption.getOrElse(firstLoss)

        val probabilities =
          Xor
            .truthTable
            .map: row =>
              Xor.predictProbability(trained.network, row.left, row.right)

        lastLoss < firstLoss * 0.1f && probabilities(0) < 0.1f && probabilities(1) > 0.9f && probabilities(
          2
        ) > 0.9f && probabilities(3) < 0.1f
      case _ => false
