package com.htmlism.magicite

import scala.util.Random

import weaver.*

object InitializationSuite extends FunSuite:
  private val fanIn  = 3
  private val fanOut = 5

  test("Xavier draws deterministically within its symmetric uniform range"):
    val limit =
      Initialization.Xavier.limit[Double](fanIn, fanOut)

    val actualA =
      Initialization
        .Xavier
        .weight[Double](fanIn, fanOut)
        .runA(Random(123))
        .value

    val actualB =
      Initialization
        .Xavier
        .weight[Double](fanIn, fanOut)
        .runA(Random(123))
        .value

    expect.all(
      actualA == actualB,
      actualA >= -limit,
      actualA < limit
    )

  test("Xavier supports Float weights"):
    val limit =
      Initialization.Xavier.limit[Float](fanIn, fanOut)

    val actualA =
      Initialization
        .Xavier
        .weight[Float](fanIn, fanOut)
        .runA(Random(123))
        .value

    val actualB =
      Initialization
        .Xavier
        .weight[Float](fanIn, fanOut)
        .runA(Random(123))
        .value

    expect.all(
      actualA == actualB,
      actualA >= -limit,
      actualA < limit
    )
