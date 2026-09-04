package com.htmlism.magicite

import scala.util.*

import weaver.*

object MatrixSuite extends FunSuite:
  test("multiplies a row-major matrix by a vector"):
    val matrix = Matrix(2, 3, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    val output = matrix.multiply(VectorN(Array(10.0, 20.0, 30.0)))

    expect.eql(Vector(140.0, 320.0), output.values.toVector)

  test("rejects matrix multiplication with an incompatible input vector"):
    val result = Try(Matrix(2, 3, Array.fill(6)(0.0)).multiply(VectorN(Array(1.0, 2.0))))

    matches(result):
      case Failure(error: IllegalArgumentException) if error.getMessage.contains("3 columns but input has 2") => success
