package com.htmlism.magicite

/** Converts a vector of class logits into a probability distribution */
object Softmax:
  /**
    * Converts logits to probabilities with the largest logit subtracted before exponentiation.
    *
    * Subtracting the maximum leaves the result unchanged while keeping the largest exponent at one, which avoids
    * overflowing on large finite logits.
    *
    * @tparam A
    *   The real-valued scalar type of the logits and probabilities
    * @tparam D
    *   The dimension of the class vector
    * @param logits
    *   One unnormalized score for each class
    */
  def probabilities[A: RealScalar as scalar, D: Dimension](logits: Vec[A, D]): Vec[A, D] =
    require(logits.size > 0, "softmax requires at least one logit")

    val maximum =
      logits.values.foldLeft(logits.values(0))(scalar.maximum)

    val exponentials =
      scalar.tabulate(logits.size): index =>
        scalar.exp(logits.values(index) - maximum)

    val denominator =
      exponentials.foldLeft(scalar.zero)(scalar.add)

    Vec[A, D](scalar.tabulate(logits.size)(index => exponentials(index) / denominator))
