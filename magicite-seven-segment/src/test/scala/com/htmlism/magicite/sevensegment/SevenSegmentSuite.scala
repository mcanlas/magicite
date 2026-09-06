package com.htmlism.magicite.sevensegment

import weaver.*

import com.htmlism.magicite.*

object SevenSegmentSuite extends FunSuite:
  test("encodes the canonical digits in digit order"):
    expect.all(
      SevenSegment.canonicalDigits.map(_.digit) == (0 to 9).toVector,
      SevenSegment.canonicalDigits.map(_.segments) == Vector(
        Vector(1, 1, 1, 1, 1, 1, 0),
        Vector(0, 1, 1, 0, 0, 0, 0),
        Vector(1, 1, 0, 1, 1, 0, 1),
        Vector(1, 1, 1, 1, 0, 0, 1),
        Vector(0, 1, 1, 0, 0, 1, 1),
        Vector(1, 0, 1, 1, 0, 1, 1),
        Vector(1, 0, 1, 1, 1, 1, 1),
        Vector(1, 1, 1, 0, 0, 0, 0),
        Vector(1, 1, 1, 1, 1, 1, 1),
        Vector(1, 1, 1, 1, 0, 1, 1)
      )
    )

  test("converts a binary row into a typed numeric input vector"):
    val zero =
      SevenSegment.encodeInputs[Double](SevenSegment.canonicalDigits(0))

    expect(zero.values.toVector == Vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0))

  test("encodes a digit target as one hot"):
    val target =
      SevenSegment.encodeTarget[Double](3)

    expect(target.values.toVector == Vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

  test("a seeded classifier learns every canonical digit"):
    val trained =
      SevenSegment.train(
        initialNetwork = SevenSegment.initialize[Double](seed = 0),
        epochCount     = 2_000,
        learningRate   = 0.1
      )

    val predictions =
      SevenSegment.canonicalDigits.map(SevenSegment.predict(trained.network, _))

    val probabilities =
      SevenSegment.canonicalDigits.map(SevenSegment.predictProbabilities(trained.network, _))

    expect.all(
      predictions == SevenSegment.canonicalDigits.map(_.digit),
      probabilities.forall(probability => math.abs(probability.values.sum - 1.0) < 1e-12)
    )
