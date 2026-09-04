package com.htmlism.magicite

/**
  * The scalar operations used by Magicite's linear algebra and activations.
  *
  * This is deliberately project-owned rather than a general-purpose numeric hierarchy: it states the exact operations
  * the current network demands. `tabulate` keeps `Vec` and `Matrix` backed by specialized primitive arrays.
  */
trait Scalar[A]:
  def zero: A
  def one: A
  def add(left: A, right: A): A
  def multiply(left: A, right: A): A
  def negate(value: A): A
  def divide(left: A, right: A): A
  def maximum(left: A, right: A): A
  def exp(value: A): A
  def tanh(value: A): A
  def tabulate(size: Int)(f: Int => A): Array[A]

object Scalar:
  given double: Scalar[Double] with
    def zero: Double                                         = 0.0
    def one: Double                                          = 1.0
    def add(left: Double, right: Double): Double             = left + right
    def multiply(left: Double, right: Double): Double        = left * right
    def negate(value: Double): Double                        = -value
    def divide(left: Double, right: Double): Double          = left / right
    def maximum(left: Double, right: Double): Double         = Math.max(left, right)
    def exp(value: Double): Double                           = Math.exp(value)
    def tanh(value: Double): Double                          = Math.tanh(value)
    def tabulate(size: Int)(f: Int => Double): Array[Double] = Array.tabulate(size)(f)

  given float: Scalar[Float] with
    def zero: Float                                        = 0.0f
    def one: Float                                         = 1.0f
    def add(left: Float, right: Float): Float              = left + right
    def multiply(left: Float, right: Float): Float         = left * right
    def negate(value: Float): Float                        = -value
    def divide(left: Float, right: Float): Float           = left / right
    def maximum(left: Float, right: Float): Float          = Math.max(left, right)
    def exp(value: Float): Float                           = Math.exp(value.toDouble).toFloat
    def tanh(value: Float): Float                          = Math.tanh(value.toDouble).toFloat
    def tabulate(size: Int)(f: Int => Float): Array[Float] = Array.tabulate(size)(f)
