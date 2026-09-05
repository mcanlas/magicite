package com.htmlism.magicite

enum Initialization:
  case Xavier

  def limit[A](fanIn: Int, fanOut: Int)(using scalar: RealScalar[A]): A =
    require(fanIn > 0, s"fanIn must be positive, but was $fanIn")
    require(fanOut > 0, s"fanOut must be positive, but was $fanOut")

    this match
      case Xavier =>
        scalar.sqrt(6.0.toScalar / (fanIn + fanOut).toDouble.toScalar)

  /** Draws one Xavier-uniform weight using the supplied layer dimensions. */
  def weight[A](fanIn: Int, fanOut: Int)(using scalar: RealScalar[A]): Rng[A] =
    this match
      case Xavier =>
        val xavierLimit = limit[A](fanIn, fanOut)

        Rng
          .nextDouble
          .map: sample =>
            (sample.toScalar * 2.0.toScalar - scalar.one) * xavierLimit
