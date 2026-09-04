package com.htmlism.magicite

final case class Matrix(rows: Int, columns: Int, values: Array[Double]):
  require(
    values.length == rows * columns,
    s"dimensions are $rows × $columns but array has ${values.length} values"
  )
