package com.htmlism.magicite.xor

import weaver.*

import com.htmlism.magicite.*

object XorSuite extends FunSuite:
  import Xor.given

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

  /*
   * Backpropagation says how changing each parameter should change the loss. This test checks that claim without
   * trusting backpropagation itself: it nudges one hidden-layer weight a tiny amount upward and downward, measures
   * the resulting loss each time, and estimates the local slope from those two measurements.
   *
   * It then compares that measured slope with the gradient produced by the complete forward-and-backward path.
   * Matching values show that the loss, sigmoid output, tanh hidden layer, and both layers' chain-rule plumbing agree.
   */
  test("matches a whole-network finite-difference gradient for a hidden weight"):
    val network =
      Xor.initialize[Double](seed = 0)

    val row =
      Xor.truthTable(2)

    val forwardPass =
      network.forward(Xor.encodeInputs[Double](row.left, row.right))

    val prediction =
      forwardPass.prediction.values(0)

    val outputGradient =
      BinaryCrossEntropy.sigmoidPreActivationDerivative(Xor.encodeTarget[Double](row.expected), prediction)

    val analyticGradient =
      network
        .backwardFromOutputPreActivation(
          forwardPass,
          Vec[Double, Xor.BooleanOutput](Array(outputGradient))
        )
        .firstHidden
        .weightGradients
        .values(0)

    val epsilon =
      1e-5

    val originalWeight =
      network.firstHidden.weights.values(0)

    val numericalGradient =
      (
        rowLoss(withFirstHiddenWeight(network, index = 0, originalWeight + epsilon), row) -
          rowLoss(withFirstHiddenWeight(network, index = 0, originalWeight - epsilon), row)
      ) / (2.0 * epsilon)

    expect(clue(math.abs(numericalGradient - analyticGradient)) < clue(1e-8))

  /*
   * A gradient is useful only if applying it moves the model in a better direction. Starting from a fixed network,
   * this test measures mean loss over all four XOR rows, performs one complete training epoch with a deliberately
   * tiny learning rate, then measures the same mean loss again.
   *
   * The update should lower loss. We use a tiny step because larger stochastic-gradient updates can overshoot or
   * temporarily make the average worse, even when their gradient calculation is correct.
   */
  test("takes a small full-epoch step downhill on mean XOR loss"):
    val initialNetwork =
      Xor.initialize[Double](seed = 0)

    val initialLoss =
      meanLoss(initialNetwork)

    val updatedNetwork =
      Xor.train(initialNetwork, epochCount = 1, learningRate = 0.001).network

    val updatedLoss =
      meanLoss(updatedNetwork)

    expect(clue(updatedLoss) < clue(initialLoss))

  private def meanLoss(network: Xor.Model[Double]): Double =
    Xor.truthTable.map(rowLoss(network, _)).sum / Xor.truthTable.size

  private def rowLoss(network: Xor.Model[Double], row: Xor.TruthTableRow): Double =
    BinaryCrossEntropy.value(
      Xor.encodeTarget[Double](row.expected),
      Xor.predictProbability(network, row.left, row.right)
    )

  private def withFirstHiddenWeight(
      network: Xor.Model[Double],
      index: Int,
      value: Double
  ): Xor.Model[Double] =
    val weights =
      network.firstHidden.weights.values.clone

    weights(index) = value

    network.copy(
      firstHidden = network.firstHidden.copy(weights = Matrix[Double, Xor.D4, Xor.XorOperands](weights))
    )
