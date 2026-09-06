package com.htmlism.magicite

/** Cross-entropy loss for a categorical target and a softmax probability vector */
object CategoricalCrossEntropy:
  /**
    * Computes the categorical cross-entropy between a target distribution and predicted class probabilities.
    *
    * A one-hot target selects the negative log-probability of its correct class. The more general distribution form is
    * useful for future label-smoothing experiments.
    *
    * @tparam A
    *   The real-valued scalar type of the target, probabilities, and loss
    * @tparam D
    *   The dimension of the class vector
    * @param target
    *   The expected class distribution
    * @param probabilities
    *   The softmax class probabilities
    */
  def value[A: RealScalar as scalar, D](target: Vec[A, D], probabilities: Vec[A, D]): A =
    target
      .values
      .indices
      .foldLeft(scalar.zero): (loss, i) =>
        loss - target.values(i) * scalar.log(probabilities.values(i))

  /** Computes the loss derivative with respect to each predicted class probability */
  def predictionDerivative[A: RealScalar as scalar, D: Dimension](
      target: Vec[A, D],
      probabilities: Vec[A, D]
  ): Vec[A, D] =
    Vec[A, D](scalar.tabulate(target.size)(i => -target.values(i) / probabilities.values(i)))

  /**
    * Computes the combined derivative of categorical cross-entropy and softmax with respect to each logit.
    *
    * This avoids materializing softmax's full Jacobian: for a target distribution that sums to one, the result is
    * simply `probabilities - target`.
    */
  def softmaxPreActivationDerivative[A: RealScalar as scalar, D: Dimension](
      target: Vec[A, D],
      probabilities: Vec[A, D]
  ): Vec[A, D] =
    Vec[A, D](scalar.tabulate(target.size)(i => probabilities.values(i) - target.values(i)))
