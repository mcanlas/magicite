package com.htmlism.magicite

/** Helpers for absolute-tolerance comparisons of values subject to floating-point rounding */
object Approx:
  /** Returns whether the absolute difference between two Double values is no more than a non-negative tolerance */
  def approximatelyEqual(left: Double, right: Double, tolerance: Double): Boolean =
    require(tolerance >= 0.0, s"tolerance must be non-negative, but was $tolerance")

    Math.abs(left - right) <= tolerance

  /** Returns whether the absolute difference between two Float values is no more than a non-negative tolerance */
  def approximatelyEqual(left: Float, right: Float, tolerance: Float): Boolean =
    require(tolerance >= 0.0f, s"tolerance must be non-negative, but was $tolerance")

    Math.abs(left - right) <= tolerance
