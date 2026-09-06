package com.htmlism.magicite

import scala.util.Random

import weaver.*

import com.htmlism.magicite.Approx.approximatelyEqual

object DenseLayerSuite extends FunSuite:
  import TestDimensions.*
  import TestDimensions.given

  test("accepts one bias for each output neuron"):
    val layer =
      DenseLayer(
        weights    = Matrix[Double, Two, Three](Array.fill(6)(0.0)),
        biases     = Vec[Double, Two](Array(0.0, 0.0)),
        activation = Activation.Tanh
      )

    expect.all(
      layer.weights.rows == 2,
      layer.weights.columns == 3,
      layer.biases.size == 2,
      layer.activation == Activation.Tanh
    )

  test("initializes Xavier weights and zero biases"):
    val layer =
      DenseLayer
        .initialize[Double, Two, Three](Initialization.Xavier, Activation.Sigmoid)
        .runA(Random(123))
        .value

    expect.all(
      layer.weights.values.exists(_ != 0.0),
      layer.biases.values.toVector == Vector(0.0, 0.0),
      layer.activation == Activation.Sigmoid
    )

  test("records pre-activations and outputs for a forward pass"):
    val layer =
      DenseLayer(
        weights    = Matrix[Double, Two, Three](Array(1.0, -1.0, 0.0, 0.0, 1.0, -1.0)),
        biases     = Vec[Double, Two](Array(0.5, -0.5)),
        activation = Activation.Tanh
      )

    val actual =
      layer.forward(Vec[Double, Three](Array(1.0, 0.5, -0.5)))

    expect.all(
      actual.input.values.toVector == Vector(1.0, 0.5, -0.5),
      actual.preActivations.values.toVector == Vector(1.0, 0.5),
      actual.outputs.values.toVector == Vector(math.tanh(1.0), math.tanh(0.5))
    )

  test("backpropagates output gradients through its activation and affine transformation"):
    val layer =
      DenseLayer(
        weights    = Matrix[Double, Two, Three](Array(1.0, -1.0, 0.0, 0.0, 1.0, -1.0)),
        biases     = Vec[Double, Two](Array(0.5, -0.5)),
        activation = Activation.Tanh
      )

    val forwardPass =
      layer.forward(Vec[Double, Three](Array(1.0, 0.5, -0.5)))

    val actual =
      layer.backward(forwardPass, Vec[Double, Two](Array(2.0, -3.0)))

    val firstGradient =
      2.0 * (1.0 - math.pow(math.tanh(1.0), 2))

    val secondGradient =
      -3.0 * (1.0 - math.pow(math.tanh(0.5), 2))

    expect.all(
      actual.preActivationGradient.values.toVector == Vector(firstGradient, secondGradient),
      actual.biasGradients.values.toVector == Vector(firstGradient, secondGradient),
      actual.weightGradients.values.toVector == Vector(
        firstGradient,
        firstGradient * 0.5,
        firstGradient * -0.5,
        secondGradient,
        secondGradient * 0.5,
        secondGradient * -0.5
      ),
      actual.inputGradient.values.toVector == Vector(
        firstGradient,
        -firstGradient + secondGradient,
        -secondGradient
      )
    )

  test("accepts the combined binary cross-entropy and sigmoid gradient"):
    val layer =
      DenseLayer(
        weights    = Matrix[Double, One, Three](Array(1.0, -1.0, 0.0)),
        biases     = Vec[Double, One](Array(0.5)),
        activation = Activation.Sigmoid
      )

    val forwardPass =
      layer.forward(Vec[Double, Three](Array(1.0, 0.5, -0.5)))

    val outputGradient =
      BinaryCrossEntropy.sigmoidPreActivationDerivative(1.0, forwardPass.outputs.values(0))

    val actual =
      layer.backwardFromPreActivation(forwardPass, Vec[Double, One](Array(outputGradient)))

    expect.all(
      actual.preActivationGradient.values.toVector == Vector(outputGradient),
      actual.weightGradients.values.toVector == Vector(outputGradient, outputGradient * 0.5, outputGradient * -0.5),
      actual.biasGradients.values.toVector == Vector(outputGradient),
      actual.inputGradient.values.toVector == Vector(outputGradient, -outputGradient, 0.0)
    )

  test("moves weights and biases opposite their gradients"):
    val layer =
      DenseLayer(
        weights    = Matrix[Double, Two, Three](Array(1.0, -1.0, 0.0, 0.0, 1.0, -1.0)),
        biases     = Vec[Double, Two](Array(0.5, -0.5)),
        activation = Activation.Tanh
      )

    val gradients =
      LayerBackwardPass(
        preActivationGradient = Vec[Double, Two](Array(2.0, -3.0)),
        weightGradients       = Matrix[Double, Two, Three](Array(2.0, 1.0, -1.0, -3.0, -1.5, 1.5)),
        biasGradients         = Vec[Double, Two](Array(2.0, -3.0)),
        inputGradient         = Vec[Double, Three](Array(0.0, 0.0, 0.0))
      )

    val actual =
      layer.updated(gradients, learningRate = 0.5)

    expect.all(
      actual.weights.values.toVector == Vector(0.0, -1.5, 0.5, 1.5, 1.75, -1.75),
      actual.biases.values.toVector == Vector(-0.5, 1.0),
      layer.weights.values.toVector == Vector(1.0, -1.0, 0.0, 0.0, 1.0, -1.0),
      layer.biases.values.toVector == Vector(0.5, -0.5)
    )

  test("matches binary cross-entropy finite differences for every weight and bias"):
    val layer =
      DenseLayer(
        weights    = Matrix[Double, One, Three](Array(1.0, -1.0, 0.0)),
        biases     = Vec[Double, One](Array(0.5)),
        activation = Activation.Sigmoid
      )

    val input =
      Vec[Double, Three](Array(1.0, 0.5, -0.5))

    val forwardPass =
      layer.forward(input)

    val outputGradient =
      BinaryCrossEntropy.sigmoidPreActivationDerivative(1.0, forwardPass.outputs.values(0))

    val analytic =
      layer.backwardFromPreActivation(forwardPass, Vec[Double, One](Array(outputGradient)))

    def loss(weights: Array[Double], biases: Array[Double]): Double =
      val prediction =
        layer
          .copy(weights = Matrix[Double, One, Three](weights), biases = Vec[Double, One](biases))
          .forward(input)
          .outputs
          .values(0)

      BinaryCrossEntropy.value(1.0, prediction)

    val epsilon =
      1e-5

    def weightGradient(index: Int): Double =
      val plus =
        layer.weights.values.clone
      plus(index) += epsilon

      val minus =
        layer.weights.values.clone
      minus(index) -= epsilon

      (loss(plus, layer.biases.values) - loss(minus, layer.biases.values)) / (2.0 * epsilon)

    def biasGradient: Double =
      val plus =
        layer.biases.values.clone
      plus(0) += epsilon

      val minus =
        layer.biases.values.clone
      minus(0) -= epsilon

      (loss(layer.weights.values, plus) - loss(layer.weights.values, minus)) / (2.0 * epsilon)

    val tolerance =
      1e-8

    expect.all(
      approximatelyEqual(weightGradient(0), analytic.weightGradients.values(0), tolerance),
      approximatelyEqual(weightGradient(1), analytic.weightGradients.values(1), tolerance),
      approximatelyEqual(weightGradient(2), analytic.weightGradients.values(2), tolerance),
      approximatelyEqual(biasGradient, analytic.biasGradients.values(0), tolerance)
    )
