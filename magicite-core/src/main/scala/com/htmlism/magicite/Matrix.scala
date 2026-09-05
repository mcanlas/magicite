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
  */
final case class Matrix[A, M, N](values: Array[A])(using
    rowDimension: Dimension[M],
    columnDimension: Dimension[N]
):
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

object Matrix:
  /** Draws row-major matrix values sequentially from the supplied initializer */
  def initialize[A, M, N](initialization: Initialization)(using
      rowDimension: Dimension[M],
      columnDimension: Dimension[N],
      scalar: RealScalar[A]
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
