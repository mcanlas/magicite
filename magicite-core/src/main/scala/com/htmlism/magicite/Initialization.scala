package com.htmlism.magicite

enum Initialization:
  case Xavier

  /**
    * Computes this initializer's symmetric weight bound for one layer
    *
    * @tparam A
    *   The real-valued scalar type of the bound
    */
  def limit[A](fanIn: Int, fanOut: Int)(using scalar: RealScalar[A]): A =
    require(fanIn > 0, s"fanIn must be positive, but was $fanIn")
    require(fanOut > 0, s"fanOut must be positive, but was $fanOut")

    this match
      case Xavier =>
        scalar.sqrt(6.0.toScalar / (fanIn + fanOut).toDouble.toScalar)

  /**
    * Draws one weight using the supplied layer dimensions
    *
    * @tparam A
    *   The real-valued scalar type of the weight
    */
  def weight[A](fanIn: Int, fanOut: Int)(using scalar: RealScalar[A]): Rng[A] =
    this match
      case Xavier =>
        val xavierLimit = limit[A](fanIn, fanOut)

        Rng
          .nextDouble
          .map: sample =>
            (sample.toScalar * 2.0.toScalar - scalar.one) * xavierLimit
