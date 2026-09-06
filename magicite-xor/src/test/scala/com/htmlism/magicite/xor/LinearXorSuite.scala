package com.htmlism.magicite.xor

import scala.util.Random

import weaver.*

import com.htmlism.magicite.*

/** A linear control for the nonlinear XOR experiment */
object LinearXorSuite extends FunSuite:
  import Xor.given

  /*
   * This model has no hidden layer: it turns the two input bits directly into one sigmoid probability.
   * Its 0.5 decision boundary is therefore one straight line. XOR's true rows sit on opposite corners,
   * so no straight line can put both true corners on one side and both false corners on the other.
   *
   * We still train it with exactly the same loss, gradient, and parameter-update machinery as the D4 model.
   * Its failure is the useful anti-example: more training cannot supply the missing nonlinear hidden features.
   */
  test("a direct sigmoid network cannot learn every XOR row"):
    val trained =
      train(
        initialNetwork = initialize(seed = 0),
        epochCount     = 5_000,
        learningRate   = 0.1
      )

    val probabilities =
      Xor
        .truthTable
        .map: row =>
          trained
            .forward(Xor.encodeInputs[Double](row.left, row.right))
            .prediction
            .values(0)

    val predictions =
      probabilities.map(Xor.decodePrediction[Double])

    println(s"Linear XOR probabilities: $probabilities")

    expect(clue(predictions) != clue(Xor.truthTable.map(_.expected)))

  private type Model =
    Network.Direct[Double, Xor.XorOperands, Xor.BooleanOutput]

  private def initialize(seed: Long): Model =
    Network
      .direct[Double, Xor.XorOperands, Xor.BooleanOutput](
        initialization   = Initialization.Xavier,
        outputActivation = Activation.Sigmoid
      )
      .runA(Random(seed))
      .value

  private def train(initialNetwork: Model, epochCount: Int, learningRate: Double): Model =
    (0 until epochCount).foldLeft(initialNetwork): (network, _) =>
      Xor
        .truthTable
        .foldLeft(network): (currentNetwork, row) =>
          val input =
            Xor.encodeInputs[Double](row.left, row.right)

          val forwardPass =
            currentNetwork.forward(input)

          val outputGradient =
            BinaryCrossEntropy.sigmoidPreActivationDerivative(
              Xor.encodeTarget[Double](row.expected),
              forwardPass.prediction.values(0)
            )

          val gradients =
            currentNetwork.backwardFromOutputPreActivation(
              forwardPass,
              Vec[Double, Xor.BooleanOutput](Array(outputGradient))
            )

          currentNetwork.updated(gradients, learningRate)
