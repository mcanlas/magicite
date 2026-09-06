package com.htmlism.magicite

/**
  * A vector whose `D` type parameter identifies its dimension.
  *
  * @tparam A
  *   The scalar type stored in each coordinate
  * @tparam D
  *   The dimension that identifies this vector's coordinates
  */
final case class Vec[A, D: Dimension as dimension](values: Array[A]):
  require(
    values.length == dimension.size,
    s"dimension has size ${dimension.size} but vector has ${values.length} values"
  )

  def size: Int =
    values.length

  /** Adds bias or gradient vectors elementwise */
  def +(that: Vec[A, D])(using Scalar[A]): Vec[A, D] =
    require(size == that.size, s"cannot add vectors with sizes $size and ${that.size}")

    val result = values.clone

    result.indices.foreach(i => result(i) = values(i) + that.values(i))

    Vec[A, D](result)

  /** Computes a neuron's weighted input sum */
  def dot(that: Vec[A, D])(using scalar: Scalar[A]): A =
    require(size == that.size, s"cannot multiply vectors with sizes $size and ${that.size}")

    values
      .indices
      .foldLeft(scalar.zero)((sum, i) => sum + values(i) * that.values(i))

  /** Applies an activation to each neuron value */
  def map(f: A => A): Vec[A, D] =
    val result = values.clone

    result.indices.foreach(i => result(i) = f(values(i)))

    Vec[A, D](result)
