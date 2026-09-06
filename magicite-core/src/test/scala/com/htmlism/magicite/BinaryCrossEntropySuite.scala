package com.htmlism.magicite

import weaver.*

import com.htmlism.magicite.Approx.approximatelyEqual

object BinaryCrossEntropySuite extends FunSuite:
  test("computes binary cross-entropy for both targets"):
    val tolerance =
      1e-12

    expect.all(
      approximatelyEqual(BinaryCrossEntropy.value(1.0, 0.8), -math.log(0.8), tolerance),
      approximatelyEqual(BinaryCrossEntropy.value(0.0, 0.8), -math.log(0.2), tolerance)
    )

  test("computes the derivative with respect to a prediction"):
    val tolerance =
      1e-12

    expect.all(
      approximatelyEqual(BinaryCrossEntropy.predictionDerivative(1.0, 0.8), -1.25, tolerance),
      approximatelyEqual(BinaryCrossEntropy.predictionDerivative(0.0, 0.8), 5.0, tolerance)
    )

  test("combines sigmoid and binary cross-entropy derivatives"):
    val tolerance =
      1e-12

    expect.all(
      approximatelyEqual(BinaryCrossEntropy.sigmoidPreActivationDerivative(1.0, 0.8), -0.2, tolerance),
      approximatelyEqual(BinaryCrossEntropy.sigmoidPreActivationDerivative(0.0, 0.8), 0.8, tolerance)
    )

  test("binary cross-entropy supports Float scalars"):
    expect.eql(
      0.8f - 1.0f,
      BinaryCrossEntropy.sigmoidPreActivationDerivative(1.0f, 0.8f)
    )
