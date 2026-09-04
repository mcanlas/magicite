package com.htmlism.magicite

final case class VectorN(values: Array[Double]):
  def size: Int =
    values.length
