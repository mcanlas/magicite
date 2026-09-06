package com.htmlism.magicite

import cats.syntax.all.*

/**
  * A network that maps input vectors tagged `I` to output vectors tagged `O`
  *
  * @tparam A
  *   The scalar type used by the network parameters
  * @tparam I
  *   The input dimension tag
  * @tparam O
  *   The output dimension tag
  */
sealed trait Network[A, I, O]

object Network:
  /**
    * A network with no hidden layers, consisting of one direct output layer
    *
    * @tparam A
    *   The scalar type used by the network parameters
    * @tparam I
    *   The input dimension tag
    * @tparam O
    *   The output dimension tag
    *
    * @param output
    *   The one dense layer that maps inputs directly to outputs
    */
  final case class Direct[A, I, O](
      output: DenseLayer[A, O, I]
  ) extends Network[A, I, O]

  /**
    * A network with one or more `D`-tagged hidden layers
    *
    * @tparam A
    *   The scalar type used by the network parameters
    * @tparam I
    *   The input dimension tag
    * @tparam D
    *   The shared hidden-layer dimension tag
    * @tparam O
    *   The output dimension tag
    *
    * @param first
    *   The first hidden layer, mapping `I` inputs to `D` outputs
    * @param middle
    *   Zero or more additional hidden layers, each mapping `D` to `D`
    * @param output
    *   The final dense layer that maps `D` hidden values to `O` outputs
    */
  final case class WithHidden[A, I, D, O](
      first: DenseLayer[A, D, I],
      middle: Vector[DenseLayer[A, D, D]],
      output: DenseLayer[A, O, D]
  ) extends Network[A, I, O]

  /**
    * Initializes a direct network for zero hidden layers, or a network whose hidden layers all use tag `D`
    *
    * @param hiddenLayerCount
    *   The number of hidden `D` layers, which must be non-negative
    * @param hiddenActivation
    *   The activation applied by each hidden `D` layer
    * @param outputActivation
    *   The activation applied by the final output layer
    *
    * @tparam A
    *   The real-valued scalar type used by the network parameters
    * @tparam I
    *   The input dimension tag
    * @tparam D
    *   The shared hidden-layer dimension tag
    * @tparam O
    *   The output dimension tag
    */
  def initialize[A: RealScalar, I: Dimension, D: Dimension, O: Dimension](
      initialization: Initialization,
      hiddenLayerCount: Int,
      hiddenActivation: Activation,
      outputActivation: Activation
  ): Rng[Network[A, I, O]] =
    require(hiddenLayerCount >= 0, s"hidden layer count must be non-negative, but was $hiddenLayerCount")

    if hiddenLayerCount == 0 then
      for output <- DenseLayer
          .initialize[A, O, I](initialization, outputActivation)
      yield Direct(output)

    else
      for
        first <- DenseLayer
          .initialize[A, D, I](initialization, hiddenActivation)

        middle <- Vector
          .fill(hiddenLayerCount - 1)(DenseLayer.initialize[A, D, D](initialization, hiddenActivation))
          .sequence

        output <- DenseLayer
          .initialize[A, O, D](initialization, outputActivation)
      yield WithHidden(first, middle, output)
