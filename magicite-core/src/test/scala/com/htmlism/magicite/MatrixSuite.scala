package com.htmlism.magicite

import scala.util.*

import weaver.*

object MatrixSuite extends FunSuite:
  import TestDimensions.*
  import TestDimensions.given

  test("multiplies a row-major matrix by a vector"):
    val matrix = Matrix[Two, Three](Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val output = matrix.multiply(Vec[Three](Array(10.0, 20.0, 30.0)))

    expect.eql(Vector(140.0, 320.0), output.values.toVector)

  test("rejects a backing array whose length disagrees with its dimensions"):
    val result = Try(Matrix[Two, Three](Array.fill(5)(0.0)))

    matches(result):
      case Failure(error: IllegalArgumentException) if error.getMessage.contains("2 x 3") => success
