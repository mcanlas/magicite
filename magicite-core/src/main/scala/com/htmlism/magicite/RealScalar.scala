package com.htmlism.magicite

/**
  * Operations available for real-valued network parameters and activations
  *
  * @tparam A
  *   The real-valued scalar type operated on by this instance
  */
trait RealScalar[A] extends Scalar[A]:
  def one: A
  def fromDouble(value: Double): A
  def negate(value: A): A
  def divide(left: A, right: A): A
  def maximum(left: A, right: A): A
  def isPositive(value: A): Boolean

  /** Computes the natural exponential, e raised to `value` */
  def exp(value: A): A

  /** Computes the natural logarithm of `value` */
  def log(value: A): A

  /** Computes the hyperbolic tangent of `value` */
  def tanh(value: A): A

  /** Computes the non-negative square root of `value` */
  def sqrt(value: A): A

  extension (left: A)
    infix def -(right: A): A =
      add(left, negate(right))

    infix def /(right: A): A =
      divide(left, right)

    def unary_- : A =
      negate(left)

  extension (value: Double)
    def toScalar: A =
      fromDouble(value)
