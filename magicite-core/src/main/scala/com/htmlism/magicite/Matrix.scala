package com.htmlism.magicite

import cats.syntax.all.*

/**
  * A row-major matrix with each row's values stored before the next row
  *
  * An M x N matrix maps an N-value input vector to an M-value output vector. For example, a 3 x 2 matrix maps two input
  * features to three neuron values
  *
  * @tparam A
  *   The scalar type stored in each matrix cell
  * @tparam M
  *   The output dimension, with one coordinate for each matrix row
  * @tparam N
  *   The input dimension, with one coordinate for each matrix column
  *
  * @param values
  *   Row-major scalar values, with one contiguous row per output coordinate
  */
final case class Matrix[A, M: Dimension as rowDimension, N: Dimension as columnDimension](values: Array[A]):
  def rows: Int =
    rowDimension.size

  def columns: Int =
    columnDimension.size

  require(
    values.length == rows * columns,
    s"dimensions are $rows x $columns but array has ${values.length} values"
  )

  /**
    * Multiplies an M x N matrix by an N-value input vector
    *
    * For example, a 3 x 2 matrix maps a two-value input vector to a three-value output vector
    *
    * @param input
    *   an N-length vector where `N == columns`
    *
    * @return
    *   an M-length vector where `M == rows`
    */
  def multiply(input: Vec[A, N])(using scalar: Scalar[A]): Vec[A, M] =
    require(
      columns == input.size,
      s"matrix has $columns columns but input has ${input.size} values"
    )

    Vec[A, M](scalar.tabulate(rows): r =>
      val rowOffset = r * columns

      (0 until columns)
        .foldLeft(scalar.zero): (sum, c) =>
          sum + values(rowOffset + c) * input.values(c))

  /**
    * Multiplies this matrix's transpose by an M-value output gradient
    *
    * @param outputGradient
    *   An M-length gradient to propagate to this matrix's N-value input side
    */
  def transposeMultiply(outputGradient: Vec[A, M])(using scalar: Scalar[A]): Vec[A, N] =
    Vec[A, N](scalar.tabulate(columns): c =>
      (0 until rows)
        .foldLeft(scalar.zero): (sum, r) =>
          sum + values(r * columns + c) * outputGradient.values(r))

object Matrix:
  /**
    * Forms an M x N matrix from every pair of left and right vector coordinates
    *
    * @tparam A
    *   The scalar type stored in the vectors and result matrix
    * @tparam M
    *   The dimension of the left vector and result rows
    * @tparam N
    *   The dimension of the right vector and result columns
    * @param left
    *   The M-length vector that supplies each result row's scale
    * @param right
    *   The N-length vector repeated across result rows
    */
  def outer[A: Scalar as scalar, M: Dimension as rowDimension, N: Dimension as columnDimension](
      left: Vec[A, M],
      right: Vec[A, N]
  ): Matrix[A, M, N] =
    Matrix[A, M, N](scalar.tabulate(rowDimension.size * columnDimension.size): i =>
      val row =
        i / columnDimension.size

      val column =
        i % columnDimension.size

      left.values(row) * right.values(column))

  /**
    * Draws row-major matrix values sequentially from the supplied initializer
    *
    * @tparam A
    *   The real-valued scalar type of each weight
    * @tparam M
    *   The output dimension, with one coordinate for each matrix row
    * @tparam N
    *   The input dimension, with one coordinate for each matrix column
    */
  def initialize[A: RealScalar as scalar, M: Dimension as rowDimension, N: Dimension as columnDimension](
      initialization: Initialization
  ): Rng[Matrix[A, M, N]] =
    val nextWeight =
      initialization.weight[A](fanIn = columnDimension.size, fanOut = rowDimension.size)

    for
      // not the same as traverse (which is xs map then sequence)
      xs <- Vector
        .fill(rowDimension.size * columnDimension.size)(nextWeight)
        .sequence
    yield
      // xs here is being used for its index lookup (hence vector)
      Matrix[A, M, N](scalar.tabulate(xs.size)(xs))
