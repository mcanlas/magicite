package com.htmlism.magicite

import weaver.*

object ScalarSuite extends FunSuite:
  test("provides infix aliases for addition and multiplication"):
    expect.eql(15.0, calculate(3.0, 4.0))

  test("provides aliases for real scalar operations"):
    expect.eql((3.0, 2.0, -6.0), calculateReal(6.0, 3.0))

  test("converts a Double to the requested scalar type"):
    expect.eql(1.25f, convert[Float](1.25))

  private def calculate[A: Scalar](left: A, right: A): A =
    left + right * left

  private def calculateReal[A: RealScalar](left: A, right: A): (A, A, A) =
    (left - right, left / right, -left)

  private def convert[A: RealScalar](value: Double): A =
    value.toScalar
