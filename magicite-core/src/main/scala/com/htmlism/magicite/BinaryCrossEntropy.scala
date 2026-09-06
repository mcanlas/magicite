package com.htmlism.magicite

/** Binary cross-entropy for one binary target and one sigmoid prediction */
object BinaryCrossEntropy:
  /**
    * Computes binary cross-entropy from a binary target and predicted probability
    *
    * @tparam A
    *   The real-valued scalar type of the target, prediction, and result
    * @param target
    *   The expected binary value, zero or one
    * @param prediction
    *   The predicted probability of the positive class
    */
  def value[A: RealScalar as scalar](target: A, prediction: A): A =
    -(
      target * scalar.log(prediction) +
        (scalar.one - target) * scalar.log(scalar.one - prediction)
    )

  /**
    * Computes the loss derivative with respect to the predicted probability
    *
    * @tparam A
    *   The real-valued scalar type of the target, prediction, and derivative
    * @param target
    *   The expected binary value, zero or one
    * @param prediction
    *   The predicted probability of the positive class
    */
  def predictionDerivative[A: RealScalar as scalar](target: A, prediction: A): A =
    -(target / prediction) + (scalar.one - target) / (scalar.one - prediction)

  /**
    * Computes the combined derivative of binary cross-entropy and a sigmoid output activation
    *
    * @tparam A
    *   The real-valued scalar type of the target, prediction, and derivative
    * @param target
    *   The expected binary value, zero or one
    * @param prediction
    *   The sigmoid output, interpreted as the positive-class probability
    */
  def sigmoidPreActivationDerivative[A: RealScalar as scalar](target: A, prediction: A): A =
    prediction - target
