package com.htmlism.magicite.xor

import weaver.*

object XorSuite extends FunSuite:
  /*
   * Experiment used to select a deterministic, comfortably convergent configuration:
   *
   * - Architecture: 2 → D4 → 1, Xavier initialization, tanh hidden activation, sigmoid output activation.
   * - The retained robustness probe checks seeds 0 through 20 at 500, 1,000, and 2,000 epochs. At rates 0.1, 0.2,
   *   and 0.3, every sampled seed passes at every epoch count for Float and Double. Rate 0.5 passes 19 of 21 seeds.
   * - At seed 0, rate 0.1, and 500 epochs, predictions are approximately 0.003, 0.981, 0.985, and 0.020 in truth-table
   *   order. The test uses that reproducible configuration while the probe remains the broader robustness check.
   *
   * This is an integration test for the whole learning path, not a search performed during ordinary test execution.
   */
  test("a seeded tanh network learns every XOR row"):
    val trained =
      Xor.train(
        initialNetwork = Xor.initialize[Double](seed = 0),
        epochCount     = 500,
        learningRate   = 0.1
      )

    val probabilities =
      Xor
        .truthTable
        .map: row =>
          Xor.predictProbability(trained.network, row.left, row.right)

    val predictions =
      Xor
        .truthTable
        .map: row =>
          Xor.predict(trained.network, row.left, row.right)

    println(s"Double XOR probabilities: $probabilities")

    trained.epochLosses match
      case firstLoss +: remainingLosses =>
        val lastLoss =
          remainingLosses.lastOption.getOrElse(firstLoss)

        expect.all(
          clue(lastLoss) < clue(firstLoss * 0.1),
          clue(probabilities(0)) < 0.1,
          clue(probabilities(1)) > 0.9,
          clue(probabilities(2)) > 0.9,
          clue(probabilities(3)) < 0.1,
          clue(predictions) == clue(Xor.truthTable.map(_.expected))
        )
      case _ => failure("expected at least one recorded epoch loss")

  test("a seeded tanh Float network learns every XOR row"):
    val trained =
      Xor.train(
        initialNetwork = Xor.initialize[Float](seed = 0),
        epochCount     = 500,
        learningRate   = 0.1f
      )

    val probabilities =
      Xor
        .truthTable
        .map: row =>
          Xor.predictProbability(trained.network, row.left, row.right)

    val predictions =
      Xor
        .truthTable
        .map: row =>
          Xor.predict(trained.network, row.left, row.right)

    println(s"Float XOR probabilities: $probabilities")

    trained.epochLosses match
      case firstLoss +: remainingLosses =>
        val lastLoss =
          remainingLosses.lastOption.getOrElse(firstLoss)

        expect.all(
          clue(lastLoss) < clue(firstLoss * 0.1f),
          clue(probabilities(0)) < 0.1f,
          clue(probabilities(1)) > 0.9f,
          clue(probabilities(2)) > 0.9f,
          clue(probabilities(3)) < 0.1f,
          clue(predictions) == clue(Xor.truthTable.map(_.expected))
        )
      case _ => failure("expected at least one recorded epoch loss")
