package com.htmlism.magicite

import scala.util.Random

import weaver.*

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
