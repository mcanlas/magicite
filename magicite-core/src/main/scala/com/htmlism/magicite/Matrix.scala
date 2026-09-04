package com.htmlism.magicite

/**
  * A row-major matrix with each row's values stored before the next row
  *
  * An M x N matrix maps an N-value input vector to an M-value output vector. For example, a 3 x 2 matrix maps two input
  * features to three neuron values
  *
  * @tparam M
  *   The output dimension, with one coordinate for each matrix row
  * @tparam N
  *   The input dimension, with one coordinate for each matrix column
  */
final case class Matrix[M, N](values: Array[Double])(using
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
  def multiply(input: Vec[N]): Vec[M] =
    require(
      columns == input.size,
      s"matrix has $columns columns but input has ${input.size} values"
    )

    Vec[M](
      Array.tabulate(rows): r =>
        val rowOffset = r * columns

        (0 until columns)
          .foldLeft(0.0): (sum, c) =>
            sum + values(rowOffset + c) * input.values(c)
    )
