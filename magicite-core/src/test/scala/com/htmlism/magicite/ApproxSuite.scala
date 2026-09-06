package com.htmlism.magicite

import scala.util.*

import weaver.*

import com.htmlism.magicite.Approx.approximatelyEqual

object ApproxSuite extends FunSuite:
  test("compares Double values within the supplied tolerance"):
    expect.all(
      approximatelyEqual(1.0, 1.1, tolerance = 0.11),
      !approximatelyEqual(1.0, 1.1, tolerance = 0.09)
    )

  test("compares Float values within the supplied tolerance"):
    expect(approximatelyEqual(1.0f, 1.1f, tolerance = 0.11f))

  test("rejects a negative Double tolerance"):
    val result =
      Try(approximatelyEqual(1.0, 1.0, tolerance = -0.1))

    matches(result):
      case Failure(error: IllegalArgumentException) if error.getMessage.contains("non-negative") => success

  test("rejects a negative Float tolerance"):
    val result =
      Try(approximatelyEqual(1.0f, 1.0f, tolerance = -0.1f))

    matches(result):
      case Failure(error: IllegalArgumentException) if error.getMessage.contains("non-negative") => success
