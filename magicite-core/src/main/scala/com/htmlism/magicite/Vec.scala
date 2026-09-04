package com.htmlism.magicite

/**
  * A vector whose `D` type parameter identifies its dimension.
  *
  * @tparam D
  *   The dimension that identifies this vector's coordinates
  */
final case class Vec[D](values: Array[Double])(using dimension: Dimension[D]):
  require(
    values.length == dimension.size,
    s"dimension has size ${dimension.size} but vector has ${values.length} values"
  )

  def size: Int =
    values.length

  /** Adds bias or gradient vectors elementwise */
  def +(that: Vec[D]): Vec[D] =
    require(size == that.size, s"cannot add vectors with sizes $size and ${that.size}")

    Vec[D](Array.tabulate(size)(i => values(i) + that.values(i)))

  /** Computes a neuron's weighted input sum */
  def dot(that: Vec[D]): Double =
    require(size == that.size, s"cannot multiply vectors with sizes $size and ${that.size}")

    values
      .indices
      .foldLeft(0.0)((sum, i) => sum + values(i) * that.values(i))

  /** Applies an activation to each neuron value */
  def map(f: Double => Double): Vec[D] =
    Vec[D](values.map(f))
