package com.htmlism.magicite

import scala.util.*

import weaver.*

object VectorNSuite extends FunSuite:
  test("adds equal-sized vectors component by component"):
    val sum = VectorN(Array(1.0, -2.0, 3.0)) + VectorN(Array(4.0, 5.0, -6.0))

    expect.eql(Vector(5.0, 3.0, -3.0), sum.values.toVector)

  test("rejects addition of differently sized vectors"):
    val result = Try(VectorN(Array(1.0)) + VectorN(Array(1.0, 2.0)))

    matches(result):
      case Failure(error: IllegalArgumentException) if error.getMessage.contains("sizes 1 and 2") => success

  test("computes a dot product"):
    val product = VectorN(Array(1.0, 2.0, 3.0)).dot(VectorN(Array(4.0, 5.0, 6.0)))

    expect.eql(32.0, product)

  test("rejects dot products of differently sized vectors"):
    val result = Try(VectorN(Array(1.0)).dot(VectorN(Array(1.0, 2.0))))

    matches(result):
      case Failure(error: IllegalArgumentException) if error.getMessage.contains("sizes 1 and 2") => success

  test("maps each component"):
    val mapped = VectorN(Array(-1.0, 0.0, 2.0)).map(_ * 2.0)

    expect.eql(Vector(-2.0, 0.0, 4.0), mapped.values.toVector)
