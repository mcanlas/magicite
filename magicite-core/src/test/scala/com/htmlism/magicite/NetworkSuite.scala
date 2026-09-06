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
      actual.outputPass.input.values.toVector == Vector(1.0, 0.5, -0.5),
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
      firstOutputs.map:
        math.tanh

    val outputPreActivation =
      additionalOutputs(0) - additionalOutputs(1) + 0.25

    expect.all(
      actual.firstHidden.preActivations.values.toVector == Vector(1.0, 0.5),
      actual.firstHidden.input.values.toVector == Vector(1.0, 0.5, -0.5),
      actual.firstHidden.outputs.values.toVector == firstOutputs,
      actual.additionalHidden.size == 1,
      actual.additionalHidden(0).input.values.toVector == firstOutputs,
      actual.additionalHidden(0).preActivations.values.toVector == firstOutputs,
      actual.additionalHidden(0).outputs.values.toVector == additionalOutputs,
      actual.outputPass.input.values.toVector == additionalOutputs,
      actual.outputPass.preActivations.values.toVector == Vector(outputPreActivation),
      actual.prediction.values.toVector == Vector(1.0 / (1.0 + math.exp(-outputPreActivation)))
    )

  test("backpropagates a direct binary-output gradient"):
    val network =
      Network.Direct(
        DenseLayer(
          weights    = Matrix[Double, One, Three](Array(1.0, -1.0, 0.0)),
          biases     = Vec[Double, One](Array(0.5)),
          activation = Activation.Sigmoid
        )
      )

    val forwardPass =
      network.forward(Vec[Double, Three](Array(1.0, 0.5, -0.5)))

    val outputGradient =
      BinaryCrossEntropy.sigmoidPreActivationDerivative(1.0, forwardPass.prediction.values(0))

    val actual =
      network.backwardFromOutputPreActivation(forwardPass, Vec[Double, One](Array(outputGradient)))

    expect.all(
      actual.outputPass.preActivationGradient.values.toVector == Vector(outputGradient),
      actual.outputPass.weightGradients.values.toVector == Vector(
        outputGradient,
        outputGradient * 0.5,
        outputGradient * -0.5
      ),
      actual.outputPass.inputGradient.values.toVector == Vector(outputGradient, -outputGradient, 0.0)
    )

  test("backpropagates through hidden layers in reverse order"):
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

    val forwardPass =
      network.forward(Vec[Double, Three](Array(1.0, 0.5, -0.5)))

    val outputGradient =
      1.5

    val actual =
      network.backward(forwardPass, Vec[Double, One](Array(outputGradient)))

    val prediction =
      forwardPass.prediction.values(0)

    val outputPreActivationGradient =
      outputGradient * prediction * (1.0 - prediction)

    val additionalOutputs =
      forwardPass.additionalHidden(0).outputs.values

    val additionalPreActivationGradient =
      Vector(
        outputPreActivationGradient * (1.0 - math.pow(additionalOutputs(0), 2)),
        -outputPreActivationGradient * (1.0 - math.pow(additionalOutputs(1), 2))
      )

    val firstOutputs =
      forwardPass.firstHidden.outputs.values

    val firstPreActivationGradient =
      Vector(
        additionalPreActivationGradient(0) * (1.0 - math.pow(firstOutputs(0), 2)),
        additionalPreActivationGradient(1) * (1.0 - math.pow(firstOutputs(1), 2))
      )

    expect.all(
      actual.outputPass.preActivationGradient.values.toVector == Vector(outputPreActivationGradient),
      actual.additionalHidden.size == 1,
      actual.additionalHidden(0).preActivationGradient.values.toVector == additionalPreActivationGradient,
      actual.firstHidden.preActivationGradient.values.toVector == firstPreActivationGradient
    )

  test("updates a direct network's output layer"):
    val network =
      Network.Direct(
        DenseLayer(
          weights    = Matrix[Double, One, Three](Array(1.0, -1.0, 0.0)),
          biases     = Vec[Double, One](Array(0.5)),
          activation = Activation.Sigmoid
        )
      )

    val gradients =
      NetworkBackwardPass.Direct(
        LayerBackwardPass(
          preActivationGradient = Vec[Double, One](Array(2.0)),
          weightGradients       = Matrix[Double, One, Three](Array(2.0, 1.0, -1.0)),
          biasGradients         = Vec[Double, One](Array(2.0)),
          inputGradient         = Vec[Double, Three](Array(0.0, 0.0, 0.0))
        )
      )

    val actual =
      network.updated(gradients, learningRate = 0.1)

    expect.all(
      actual.output.weights.values.toVector == Vector(0.8, -1.1, 0.1),
      actual.output.biases.values.toVector == Vector(0.3)
    )

  test("updates every layer of a hidden network"):
    val network =
      Network.WithHidden(
        firstHidden = DenseLayer(
          weights    = Matrix[Double, Two, Three](Array.fill(6)(10.0)),
          biases     = Vec[Double, Two](Array(10.0, 10.0)),
          activation = Activation.Tanh
        ),
        additionalHidden = Vector(
          DenseLayer(
            weights    = Matrix[Double, Two, Two](Array.fill(4)(20.0)),
            biases     = Vec[Double, Two](Array(20.0, 20.0)),
            activation = Activation.Tanh
          )
        ),
        output = DenseLayer(
          weights    = Matrix[Double, One, Two](Array.fill(2)(30.0)),
          biases     = Vec[Double, One](Array(30.0)),
          activation = Activation.Sigmoid
        )
      )

    val gradients =
      NetworkBackwardPass.WithHidden(
        firstHidden = LayerBackwardPass(
          preActivationGradient = Vec[Double, Two](Array(3.0, 3.0)),
          weightGradients       = Matrix[Double, Two, Three](Array.fill(6)(2.0)),
          biasGradients         = Vec[Double, Two](Array(3.0, 3.0)),
          inputGradient         = Vec[Double, Three](Array.fill(3)(0.0))
        ),
        additionalHidden = Vector(
          LayerBackwardPass(
            preActivationGradient = Vec[Double, Two](Array(5.0, 5.0)),
            weightGradients       = Matrix[Double, Two, Two](Array.fill(4)(4.0)),
            biasGradients         = Vec[Double, Two](Array(5.0, 5.0)),
            inputGradient         = Vec[Double, Two](Array(0.0, 0.0))
          )
        ),
        outputPass = LayerBackwardPass(
          preActivationGradient = Vec[Double, One](Array(7.0)),
          weightGradients       = Matrix[Double, One, Two](Array.fill(2)(6.0)),
          biasGradients         = Vec[Double, One](Array(7.0)),
          inputGradient         = Vec[Double, Two](Array(0.0, 0.0))
        )
      )

    val actual =
      network.updated(gradients, learningRate = 0.1)

    expect.all(
      actual.firstHidden.weights.values.toVector == Vector.fill(6)(9.8),
      actual.firstHidden.biases.values.toVector == Vector.fill(2)(9.7),
      actual.additionalHidden(0).weights.values.toVector == Vector.fill(4)(19.6),
      actual.additionalHidden(0).biases.values.toVector == Vector.fill(2)(19.5),
      actual.output.weights.values.toVector == Vector.fill(2)(29.4),
      actual.output.biases.values.toVector == Vector(29.3)
    )
