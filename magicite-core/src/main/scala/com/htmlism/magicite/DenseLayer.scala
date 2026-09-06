package com.htmlism.magicite

/**
  * A layer of neurons that is fully connected to the previous layer
  *
  * @tparam A
  *   The scalar type used by the layer's weights and biases
  * @tparam M
  *   The output dimension, with one coordinate for each output neuron
  * @tparam N
  *   The input dimension, with one coordinate for each input feature
  *
  * @param weights
  *   The row-major weight matrix, the linear part of the affine transformation from `N` inputs to `M` outputs
  * @param biases
  *   One additive bias value for each output neuron
  * @param activation
  *   The scalar activation applied to each pre-activation
  */
final case class DenseLayer[A, M: Dimension, N: Dimension](
    weights: Matrix[A, M, N],
    biases: Vec[A, M],
    activation: Activation
):
  require(
    biases.size == weights.rows,
    s"layer has ${weights.rows} output rows but ${biases.size} biases"
  )

  /** Runs the layer's affine transformation and activation for one input vector */
  def forward(input: Vec[A, N])(using RealScalar[A]): LayerForwardPass[A, N, M] =
    val preActivations =
      weights.multiply(input) + biases

    LayerForwardPass(
      input          = input,
      preActivations = preActivations,
      outputs        = preActivations.map(activation.apply)
    )

  /**
    * Backpropagates a loss gradient with respect to this layer's activated outputs
    *
    * @param forwardPass
    *   The cached values from this layer's forward invocation
    * @param outputGradient
    *   The loss gradient with respect to the activated outputs
    */
  def backward(
      forwardPass: LayerForwardPass[A, N, M],
      outputGradient: Vec[A, M]
  )(using scalar: RealScalar[A]): LayerBackwardPass[A, N, M] =
    val preActivationGradient =
      Vec[A, M](scalar.tabulate(outputGradient.size): index =>
        outputGradient.values(index) * activation.derivative(
          forwardPass.preActivations.values(index),
          forwardPass.outputs.values(index)
        ))

    backwardFromPreActivation(forwardPass, preActivationGradient)

  /**
    * Backpropagates a loss gradient that is already with respect to affine outputs
    *
    * This is useful for binary cross-entropy paired with a sigmoid output, whose combined gradient is
    * `prediction - target`.
    *
    * @param forwardPass
    *   The cached values from this layer's forward invocation
    * @param preActivationGradient
    *   The loss gradient with respect to the affine outputs
    */
  def backwardFromPreActivation(
      forwardPass: LayerForwardPass[A, N, M],
      preActivationGradient: Vec[A, M]
  )(using Scalar[A]): LayerBackwardPass[A, N, M] =
    LayerBackwardPass(
      preActivationGradient = preActivationGradient,
      weightGradients       = Matrix.outer(preActivationGradient, forwardPass.input),
      biasGradients         = preActivationGradient,
      inputGradient         = weights.transposeMultiply(preActivationGradient)
    )

  /**
    * Returns a copy with its weights and biases moved opposite their gradients
    *
    * @param gradients
    *   Gradients produced for this layer by backpropagation
    * @param learningRate
    *   The positive scale of this gradient-descent step
    */
  def updated(
      gradients: LayerBackwardPass[A, N, M],
      learningRate: A
  )(using scalar: RealScalar[A]): DenseLayer[A, M, N] =
    DenseLayer(
      weights = Matrix[A, M, N](scalar.tabulate(weights.values.length): index =>
        weights.values(index) - learningRate * gradients.weightGradients.values(index)),
      biases = Vec[A, M](scalar.tabulate(biases.values.length): index =>
        biases.values(index) - learningRate * gradients.biasGradients.values(index)),
      activation = activation
    )

object DenseLayer:
  /**
    * Draws weights and sets biases to zero
    *
    * @tparam A
    *   The real-valued scalar type used by the layer
    * @tparam M
    *   The output dimension, with one coordinate for each output neuron
    * @tparam N
    *   The input dimension, with one coordinate for each input feature
    */
  def initialize[A: RealScalar as scalar, M: Dimension as rowDimension, N: Dimension](
      initialization: Initialization,
      activation: Activation
  ): Rng[DenseLayer[A, M, N]] =
    for weights <- Matrix.initialize[A, M, N](initialization)
    yield DenseLayer(
      weights    = weights,
      biases     = Vec[A, M](scalar.tabulate(rowDimension.size)(_ => scalar.zero)),
      activation = activation
    )
