package com.htmlism.magicite

import weaver.*

object DenseLayerSuite extends FunSuite:
  import TestDimensions.*
  import TestDimensions.given

  test("accepts one bias for each output neuron"):
    val layer =
      DenseLayer(
        weights    = Matrix[Two, Three](Array.fill(6)(0.0)),
        biases     = Vec[Two](Array(0.0, 0.0)),
        activation = Activation.Tanh
      )

    expect.all(
      layer.weights.rows == 2,
      layer.weights.columns == 3,
      layer.biases.size == 2,
      layer.activation == Activation.Tanh
    )
