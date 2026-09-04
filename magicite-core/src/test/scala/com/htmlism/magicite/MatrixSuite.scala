package com.htmlism.magicite

import scala.util.*

import weaver.*

object MatrixSuite extends FunSuite:
  import TestDimensions.*
  import TestDimensions.given

  test("multiplies a row-major matrix by a vector"):
    val matrix = Matrix[Double, Two, Three](Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val output = matrix.multiply(Vec[Double, Three](Array(10.0, 20.0, 30.0)))

    expect.eql(Vector(140.0, 320.0), output.values.toVector)

  test("rejects a backing array whose length disagrees with its dimensions"):
    val result = Try(Matrix[Double, Two, Three](Array.fill(5)(0.0)))

    matches(result):
      case Failure(error: IllegalArgumentException) if error.getMessage.contains("2 x 3") => success

  test("multiplies Float matrices and vectors"):
    val matrix = Matrix[Float, Two, Three](Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
    val output = matrix.multiply(Vec[Float, Three](Array(10.0f, 20.0f, 30.0f)))

    expect.eql(Vector(140.0f, 320.0f), output.values.toVector)
