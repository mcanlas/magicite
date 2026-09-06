package com.htmlism.magicite

import scala.util.Random

import weaver.*

object NetworkSuite extends FunSuite:
  import TestDimensions.*
  import TestDimensions.given

  test("initializes a direct network when the hidden layer count is zero"):
    val actual =
      Network
        .initialize[Double, Three, Two, One](
          Initialization.Xavier,
          hiddenLayerCount = 0,
          hiddenActivation = Activation.Tanh,
          outputActivation = Activation.Sigmoid
        )
        .runA(Random(123))
        .value

    actual match
      case Network.Direct(output) =>
        expect.all(
          output.weights.rows == 1,
          output.weights.columns == 3,
          output.biases.values.toVector == Vector(0.0),
          output.activation == Activation.Sigmoid
        )
      case _ => failure("expected a direct network")

  test("initializes the requested number of D-tagged hidden layers"):
    val actual =
      Network
        .initialize[Double, Three, Two, One](
          Initialization.Xavier,
          hiddenLayerCount = 2,
          hiddenActivation = Activation.Tanh,
          outputActivation = Activation.Sigmoid
        )
        .runA(Random(123))
        .value

    actual match
      case Network.WithHidden(first, Vector(middle), output) =>
        expect.all(
          first.weights.rows == 2,
          first.weights.columns == 3,
          first.activation == Activation.Tanh,
          middle.weights.rows == 2,
          middle.weights.columns == 2,
          middle.activation == Activation.Tanh,
          output.weights.rows == 1,
          output.weights.columns == 2,
          output.activation == Activation.Sigmoid
        )
      case Network.WithHidden(_, _, _) => failure("expected one middle hidden layer")
      case _                           => failure("expected a network with hidden layers")
