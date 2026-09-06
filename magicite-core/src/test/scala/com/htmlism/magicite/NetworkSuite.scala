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

  test("runs a direct network forward"):
    val network =
      Network.Direct(
        DenseLayer(
          weights    = Matrix[Double, One, Three](Array(1.0, -1.0, 0.0)),
          biases     = Vec[Double, One](Array(0.5)),
          activation = Activation.Sigmoid
        )
      )

    val input =
      Vec[Double, Three](Array(1.0, 0.5, -0.5))

    val actual =
      network.forward(input)

    expect.all(
      actual.input.values.toVector == Vector(1.0, 0.5, -0.5),
      actual.outputPass.preActivations.values.toVector == Vector(1.0),
      actual.prediction.values.toVector == Vector(1.0 / (1.0 + math.exp(-1.0)))
    )

  test("runs a network with hidden layers forward"):
    val network =
      Network.WithHidden(
        firstHidden = DenseLayer(
          weights    = Matrix[Double, Two, Three](Array(1.0, -1.0, 0.0, 0.0, 1.0, -1.0)),
          biases     = Vec[Double, Two](Array(0.5, -0.5)),
          activation = Activation.Tanh
        ),
        additionalHidden = Vector(
          DenseLayer(
            weights    = Matrix[Double, Two, Two](Array(1.0, 0.0, 0.0, 1.0)),
            biases     = Vec[Double, Two](Array(0.0, 0.0)),
            activation = Activation.Tanh
          )
        ),
        output = DenseLayer(
          weights    = Matrix[Double, One, Two](Array(1.0, -1.0)),
          biases     = Vec[Double, One](Array(0.25)),
          activation = Activation.Sigmoid
        )
      )

    val actual =
      network.forward(Vec[Double, Three](Array(1.0, 0.5, -0.5)))

    val firstOutputs =
      Vector(math.tanh(1.0), math.tanh(0.5))

    val additionalOutputs =
      firstOutputs.map(math.tanh)

    val outputPreActivation =
      additionalOutputs(0) - additionalOutputs(1) + 0.25

    expect.all(
      actual.firstHidden.preActivations.values.toVector == Vector(1.0, 0.5),
      actual.firstHidden.outputs.values.toVector == firstOutputs,
      actual.additionalHidden.size == 1,
      actual.additionalHidden(0).preActivations.values.toVector == firstOutputs,
      actual.additionalHidden(0).outputs.values.toVector == additionalOutputs,
      actual.outputPass.preActivations.values.toVector == Vector(outputPreActivation),
      actual.prediction.values.toVector == Vector(1.0 / (1.0 + math.exp(-outputPreActivation)))
    )
