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
