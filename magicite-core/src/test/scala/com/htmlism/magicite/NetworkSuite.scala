package com.htmlism.magicite

import scala.util.*

import weaver.*

object NetworkSuite extends FunSuite:
  import TestDimensions.*
  import TestDimensions.given

  test("initializes a direct network without a hidden tag"):
    val actual =
      Network
        .direct[Double, Three, One](
          Initialization.Xavier,
          outputActivation = Activation.Sigmoid
        )
        .runA(Random(123))
        .value

    expect.all(
      actual.output.weights.rows == 1,
      actual.output.weights.columns == 3,
      actual.output.biases.values.toVector == Vector(0.0),
      actual.output.activation == Activation.Sigmoid
    )

  test("initializes the requested total number of D-tagged hidden layers"):
    val actual =
      Network
        .withHidden[Double, Three, Two, One](
          Initialization.Xavier,
          hiddenLayerCount = 2,
          hiddenActivation = Activation.Tanh,
          outputActivation = Activation.Sigmoid
        )
        .runA(Random(123))
        .value

    actual.additionalHidden match
      case Vector(additionalHidden) =>
        expect.all(
          actual.firstHidden.weights.rows == 2,
          actual.firstHidden.weights.columns == 3,
          actual.firstHidden.activation == Activation.Tanh,
          additionalHidden.weights.rows == 2,
          additionalHidden.weights.columns == 2,
          additionalHidden.activation == Activation.Tanh,
          actual.output.weights.rows == 1,
          actual.output.weights.columns == 2,
          actual.output.activation == Activation.Sigmoid
        )
      case _ => failure("expected one middle hidden layer")

  test("initializes an I-to-D-to-O network with one hidden layer"):
    val actual =
      Network
        .withHidden[Double, Three, Two, One](
          Initialization.Xavier,
          hiddenLayerCount = 1,
          hiddenActivation = Activation.Tanh,
          outputActivation = Activation.Sigmoid
        )
        .runA(Random(123))
        .value

    expect.all(
      actual.firstHidden.weights.rows == 2,
      actual.firstHidden.weights.columns == 3,
      actual.additionalHidden.isEmpty,
      actual.output.weights.rows == 1,
      actual.output.weights.columns == 2
    )

  test("rejects zero hidden layers because direct networks use Network.direct"):
    val actual =
      Try(
        Network.withHidden[Double, Three, Two, One](
          Initialization.Xavier,
          hiddenLayerCount = 0,
          hiddenActivation = Activation.Tanh,
          outputActivation = Activation.Sigmoid
        )
      )

    matches(actual):
      case Failure(error: IllegalArgumentException) if error.getMessage.contains("must be positive") => success
