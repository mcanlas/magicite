package com.htmlism.magicite.xor

import weaver.*

object XorSuite extends FunSuite:
  /*
   * Experiment used to select a deterministic, comfortably convergent configuration:
   *
   * - Architecture: 2 → D2 → 1, Xavier initialization, tanh hidden activation, sigmoid output activation.
   * - First probe: seeds 0 through 20 with rates 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, and 1.0 for 10,000 epochs.
   *   Seed 0 and rate 0.1 converged strongly.
   * - Epoch sweep with that seed and rate: 300 epochs cleared the 0.1/0.9 prediction thresholds, but narrowly missed
   *   the 90% loss-reduction assertion. 400 cleared every assertion. 500 retains useful margin.
   * - At 500 epochs, predictions are approximately 0.020, 0.966, 0.967, and 0.017 in truth-table order.
   *
   * This is an integration test for the whole learning path, not a search performed during ordinary test execution.
   */
  test("a seeded tanh network learns every XOR row"):
    val trained =
      Xor.train(
        initialNetwork = Xor.initialize(seed = 0),
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

    trained.epochLosses match
      case firstLoss +: remainingLosses =>
        val lastLoss =
          remainingLosses.lastOption.getOrElse(firstLoss)

        expect.all(
          lastLoss < firstLoss * 0.1,
          probabilities(0) < 0.1,
          probabilities(1) > 0.9,
          probabilities(2) > 0.9,
          probabilities(3) < 0.1,
          predictions == Xor.truthTable.map(_.expected)
        )
      case _ => failure("expected at least one recorded epoch loss")
