package com.htmlism.magicite.sevensegment

import weaver.*

import com.htmlism.magicite.*
import com.htmlism.magicite.Approx.approximatelyEqual

/** Tests for shared canonical-display data and the generic multiclass classifier */
object SevenSegmentSuite extends FunSuite:
  test("encodes the canonical digits in digit order"):
    val digits =
      SevenSegment
        .canonicalDigits
        .map:
          _.digit

    val segments =
      SevenSegment
        .canonicalDigits
        .map:
          _.segments

    expect.all(
      digits == (0 to 9).toVector,
      segments == Vector(
        SegmentState(1, 1, 1, 1, 1, 1, 0),
        SegmentState(0, 1, 1, 0, 0, 0, 0),
        SegmentState(1, 1, 0, 1, 1, 0, 1),
        SegmentState(1, 1, 1, 1, 0, 0, 1),
        SegmentState(0, 1, 1, 0, 0, 1, 1),
        SegmentState(1, 0, 1, 1, 0, 1, 1),
        SegmentState(1, 0, 1, 1, 1, 1, 1),
        SegmentState(1, 1, 1, 0, 0, 0, 0),
        SegmentState(1, 1, 1, 1, 1, 1, 1),
        SegmentState(1, 1, 1, 1, 0, 1, 1)
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
      SevenSegment
        .canonicalDigits
        .map:
          SevenSegment.predict(trained.network, _)

    val probabilities =
      SevenSegment
        .canonicalDigits
        .map:
          SevenSegment.predictProbabilities(trained.network, _)

    val expectedDigits =
      SevenSegment
        .canonicalDigits
        .map:
          _.digit

    expect.all(
      predictions == expectedDigits,
      probabilities.forall(probability => approximatelyEqual(probability.values.sum, 1.0, tolerance = 1e-12))
    )
