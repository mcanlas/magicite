package com.htmlism.magicite

/**
  * Runtime size evidence for a phantom shape tag.
  *
  * `A` is a phantom shape tag: it exists only in the type system. It makes a vector of input features distinct from,
  * for example, a vector of hidden units even when both have the same numeric size.
  *
  * @tparam A
  *   The nominal shape tag whose runtime size this evidence supplies
  */
trait Dimension[A]:
  def size: Int

object Dimension:
  /**
    * Creates size evidence for a phantom shape tag.
    *
    * @tparam A
    *   The nominal shape tag whose runtime size this evidence supplies
    *
    * @param dimensionSize
    *   The non-negative runtime size of `A`
    */
  def apply[A](dimensionSize: Int): Dimension[A] =
    require(dimensionSize >= 0, s"dimension size must be non-negative, but was $dimensionSize")

    new Dimension[A]:
      val size: Int = dimensionSize
