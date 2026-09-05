package com.htmlism.magicite

/** Operations available for real-valued network parameters and activations. */
trait RealScalar[A] extends Scalar[A]:
  def one: A
  def fromDouble(value: Double): A
  def negate(value: A): A
  def divide(left: A, right: A): A
  def maximum(left: A, right: A): A
  def exp(value: A): A
  def tanh(value: A): A
  def sqrt(value: A): A

  extension (value: Double)
    def toScalar: A =
      fromDouble(value)
