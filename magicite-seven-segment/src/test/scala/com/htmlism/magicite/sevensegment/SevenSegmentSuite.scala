package com.htmlism.magicite.sevensegment

import weaver.*

import com.htmlism.magicite.*

object SevenSegmentSuite extends FunSuite:
  test("encodes the canonical digits in digit order"):
    expect.all(
      SevenSegment.truthTable.map(_.digit) == (0 to 9).toVector,
      SevenSegment.truthTable.map(_.segments) == Vector(
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
      SevenSegment.encodeInputs[Double](SevenSegment.truthTable(0))

    expect(zero.values.toVector == Vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0))
