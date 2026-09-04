package com.htmlism.magicite

import scala.util.Failure
import scala.util.Try

import weaver.*

object DenseLayerSuite extends FunSuite:
  test("accepts one bias for each output neuron"):
    val layer =
      DenseLayer(
        weights    = Matrix(2, 3, Array.fill(6)(0.0)),
        biases     = VectorN(Array(0.0, 0.0)),
        activation = Activation.Tanh
      )

    expect.all(
      layer.weights.rows == 2,
      layer.weights.columns == 3,
      layer.biases.size == 2,
      layer.activation == Activation.Tanh
    )

  test("rejects a bias vector with the wrong output size"):
    val result =
      Try(
        DenseLayer(
          weights    = Matrix(2, 3, Array.fill(6)(0.0)),
          biases     = VectorN(Array(0.0)),
          activation = Activation.Identity
        )
      )

    matches(result):
      case Failure(error: IllegalArgumentException) if error.getMessage.contains("2 output rows") => success
