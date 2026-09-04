package com.htmlism.magicite

enum Initialization:
  case Xavier

  def limit[A](fanIn: Int, fanOut: Int)(using scalar: RealScalar[A]): A =
    require(fanIn > 0, s"fanIn must be positive, but was $fanIn")
    require(fanOut > 0, s"fanOut must be positive, but was $fanOut")

    this match
      case Xavier =>
        scalar.sqrt(scalar.divide(scalar.fromDouble(6.0), scalar.fromDouble((fanIn + fanOut).toDouble)))

  /** Draws one Xavier-uniform weight using the supplied layer dimensions. */
  def weight[A](fanIn: Int, fanOut: Int)(using scalar: RealScalar[A]): Rng[A] =
    this match
      case Xavier =>
        val xavierLimit = limit[A](fanIn, fanOut)

        Rng
          .nextDouble
          .map: sample =>
            scalar.multiply(
              scalar.add(scalar.multiply(scalar.fromDouble(sample), scalar.fromDouble(2.0)), scalar.negate(scalar.one)),
              xavierLimit
            )
