package com.htmlism.magicite

import scala.util.*

import weaver.*

object VecSuite extends FunSuite:
  import TestDimensions.*
  import TestDimensions.given

  test("adds equal-sized vectors component by component"):
    val sum = Vec[Double, Three](Array(1.0, -2.0, 3.0)) + Vec[Double, Three](Array(4.0, 5.0, -6.0))

    expect.eql(Vector(5.0, 3.0, -3.0), sum.values.toVector)

  test("rejects a backing array whose length disagrees with its dimension"):
    val result = Try(Vec[Double, Two](Array(1.0)))

    matches(result):
      case Failure(error: IllegalArgumentException) if error.getMessage.contains("size 2") => success

  test("computes a dot product"):
    val product = Vec[Double, Three](Array(1.0, 2.0, 3.0)).dot(Vec[Double, Three](Array(4.0, 5.0, 6.0)))

    expect.eql(32.0, product)

  test("maps each component"):
    val mapped = Vec[Double, Three](Array(-1.0, 0.0, 2.0)).map(_ * 2.0)

    expect.eql(Vector(-2.0, 0.0, 4.0), mapped.values.toVector)
