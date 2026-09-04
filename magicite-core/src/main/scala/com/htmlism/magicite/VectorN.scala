package com.htmlism.magicite

/** A fixed-size vector of neuron values */
final case class VectorN(values: Array[Double]):
  def size: Int =
    values.length

  /** Adds bias or gradient vectors elementwise */
  def +(that: VectorN): VectorN =
    require(size == that.size, s"cannot add vectors with sizes $size and ${that.size}")

    VectorN(Array.tabulate(size)(i => values(i) + that.values(i)))

  /** Computes a neuron's weighted input sum */
  def dot(that: VectorN): Double =
    require(size == that.size, s"cannot multiply vectors with sizes $size and ${that.size}")

    values
      .indices
      .foldLeft(0.0)((sum, i) => sum + values(i) * that.values(i))

  /** Applies an activation to each neuron value */
  def map(f: Double => Double): VectorN =
    VectorN(values.map(f))
