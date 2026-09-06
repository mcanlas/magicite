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
final case class DenseLayer[A, M, N](
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
