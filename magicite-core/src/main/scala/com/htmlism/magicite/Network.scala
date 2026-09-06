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
sealed trait Network[A, I, O]:
  /** Runs every layer for one input vector and records the intermediate values */
  def forward(input: Vec[A, I])(using RealScalar[A]): NetworkForwardPass[A, I, O]

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
  ) extends Network[A, I, O]:
    def forward(input: Vec[A, I])(using RealScalar[A]): NetworkForwardPass.Direct[A, I, O] =
      NetworkForwardPass.Direct(input, output.forward(input))

    /**
      * Backpropagates a loss gradient with respect to this network's activated outputs
      *
      * @param forwardPass
      *   The cached values from this network's forward invocation
      * @param outputGradient
      *   The loss gradient with respect to the activated network outputs
      */
    def backward(
        forwardPass: NetworkForwardPass.Direct[A, I, O],
        outputGradient: Vec[A, O]
    )(using RealScalar[A]): NetworkBackwardPass.Direct[A, I, O] =
      NetworkBackwardPass.Direct(output.backward(forwardPass.outputPass, outputGradient))

    /**
      * Backpropagates a loss gradient that is already with respect to output affine values
      *
      * @param forwardPass
      *   The cached values from this network's forward invocation
      * @param outputPreActivationGradient
      *   The loss gradient with respect to the output layer's affine values
      */
    def backwardFromOutputPreActivation(
        forwardPass: NetworkForwardPass.Direct[A, I, O],
        outputPreActivationGradient: Vec[A, O]
    )(using Scalar[A]): NetworkBackwardPass.Direct[A, I, O] =
      NetworkBackwardPass.Direct(
        output.backwardFromPreActivation(forwardPass.outputPass, outputPreActivationGradient)
      )

    /**
      * Returns a copy with its output-layer parameters moved opposite their gradients
      *
      * @param gradients
      *   Gradients produced for this network by backpropagation
      * @param learningRate
      *   The positive scale of this gradient-descent step
      */
    def updated(
        gradients: NetworkBackwardPass.Direct[A, I, O],
        learningRate: A
    )(using RealScalar[A]): Direct[A, I, O] =
      Direct(output.updated(gradients.outputPass, learningRate))

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
    * @param firstHidden
    *   The first hidden layer, mapping `I` inputs to `D` outputs
    * @param additionalHidden
    *   Zero or more additional hidden layers, each mapping `D` to `D`
    * @param output
    *   The final dense layer that maps `D` hidden values to `O` outputs
    */
  final case class WithHidden[A, I, D, O](
      firstHidden: DenseLayer[A, D, I],
      additionalHidden: Vector[DenseLayer[A, D, D]],
      output: DenseLayer[A, O, D]
  ) extends Network[A, I, O]:
    def forward(input: Vec[A, I])(using RealScalar[A]): NetworkForwardPass.WithHidden[A, I, D, O] =
      val firstForwardPass =
        firstHidden.forward(input)

      val (additionalForwardPasses, finalHiddenOutput) =
        additionalHidden.foldLeft(Vector.empty[LayerForwardPass[A, D, D]] -> firstForwardPass.outputs):
          case ((forwardPasses, hiddenOutput), layer) =>
            val forwardPass =
              layer.forward(hiddenOutput)

            (forwardPasses :+ forwardPass) -> forwardPass.outputs

      NetworkForwardPass.WithHidden(
        input            = input,
        firstHidden      = firstForwardPass,
        additionalHidden = additionalForwardPasses,
        outputPass       = output.forward(finalHiddenOutput)
      )

    /**
      * Backpropagates a loss gradient with respect to this network's activated outputs
      *
      * @param forwardPass
      *   The cached values from this network's forward invocation
      * @param outputGradient
      *   The loss gradient with respect to the activated network outputs
      */
    def backward(
        forwardPass: NetworkForwardPass.WithHidden[A, I, D, O],
        outputGradient: Vec[A, O]
    )(using RealScalar[A]): NetworkBackwardPass.WithHidden[A, I, D, O] =
      val outputBackwardPass =
        output.backward(forwardPass.outputPass, outputGradient)

      backwardFromOutputBackwardPass(forwardPass, outputBackwardPass)

    /**
      * Backpropagates a loss gradient that is already with respect to output affine values
      *
      * @param forwardPass
      *   The cached values from this network's forward invocation
      * @param outputPreActivationGradient
      *   The loss gradient with respect to the output layer's affine values
      */
    def backwardFromOutputPreActivation(
        forwardPass: NetworkForwardPass.WithHidden[A, I, D, O],
        outputPreActivationGradient: Vec[A, O]
    )(using RealScalar[A]): NetworkBackwardPass.WithHidden[A, I, D, O] =
      val outputBackwardPass =
        output.backwardFromPreActivation(forwardPass.outputPass, outputPreActivationGradient)

      backwardFromOutputBackwardPass(forwardPass, outputBackwardPass)

    private def backwardFromOutputBackwardPass(
        forwardPass: NetworkForwardPass.WithHidden[A, I, D, O],
        outputBackwardPass: LayerBackwardPass[A, D, O]
    )(using RealScalar[A]): NetworkBackwardPass.WithHidden[A, I, D, O] =
      require(
        additionalHidden.size == forwardPass.additionalHidden.size,
        s"network has ${additionalHidden.size} additional hidden layers but forward pass has ${forwardPass.additionalHidden.size}"
      )

      val (additionalBackwardPasses, firstHiddenOutputGradient) =
        additionalHidden
          .zip(forwardPass.additionalHidden)
          .reverse
          .foldLeft(Vector.empty[LayerBackwardPass[A, D, D]] -> outputBackwardPass.inputGradient):
            case ((backwardPasses, outputGradient), (layer, layerForwardPass)) =>
              val backwardPass =
                layer.backward(layerForwardPass, outputGradient)

              (backwardPass +: backwardPasses) -> backwardPass.inputGradient

      NetworkBackwardPass.WithHidden(
        firstHidden      = firstHidden.backward(forwardPass.firstHidden, firstHiddenOutputGradient),
        additionalHidden = additionalBackwardPasses,
        outputPass       = outputBackwardPass
      )

    /**
      * Returns a copy with every layer's parameters moved opposite their gradients
      *
      * @param gradients
      *   Gradients produced for this network by backpropagation
      * @param learningRate
      *   The positive scale of this gradient-descent step
      */
    def updated(
        gradients: NetworkBackwardPass.WithHidden[A, I, D, O],
        learningRate: A
    )(using RealScalar[A]): WithHidden[A, I, D, O] =
      require(
        additionalHidden.size == gradients.additionalHidden.size,
        s"network has ${additionalHidden.size} additional hidden layers but gradients have ${gradients.additionalHidden.size}"
      )

      val updatedAdditionalHidden =
        additionalHidden
          .zip(gradients.additionalHidden)
          .map:
            case (layer, layerGradients) => layer.updated(layerGradients, learningRate)

      WithHidden(
        firstHidden      = firstHidden.updated(gradients.firstHidden, learningRate),
        additionalHidden = updatedAdditionalHidden,
        output           = output.updated(gradients.outputPass, learningRate)
      )

  /**
    * Initializes a network with no hidden layers
    *
    * @param initialization
    *   The policy that draws the output-layer weights
    * @param outputActivation
    *   The activation applied by the final output layer
    *
    * @tparam A
    *   The real-valued scalar type used by the network parameters
    * @tparam I
    *   The input dimension tag
    * @tparam O
    *   The output dimension tag
    */
  def direct[A: RealScalar, I: Dimension, O: Dimension](
      initialization: Initialization,
      outputActivation: Activation
  ): Rng[Direct[A, I, O]] =
    for output <- DenseLayer
        .initialize[A, O, I](initialization, outputActivation)
    yield Direct(output)

  /**
    * Initializes a network with an `I → D → O` path and optional additional `D → D` hidden layers
    *
    * @param initialization
    *   The policy that draws all layer weights
    * @param hiddenLayerCount
    *   The total number of hidden `D` layers, which must be positive
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
  def withHidden[A: RealScalar, I: Dimension, D: Dimension, O: Dimension](
      initialization: Initialization,
      hiddenLayerCount: Int,
      hiddenActivation: Activation,
      outputActivation: Activation
  ): Rng[WithHidden[A, I, D, O]] =
    require(
      hiddenLayerCount > 0,
      s"hidden layer count must be positive, but was $hiddenLayerCount"
    )

    for
      firstHidden <- DenseLayer
        .initialize[A, D, I](initialization, hiddenActivation)

      additionalHidden <- Vector
        .fill(hiddenLayerCount - 1)(DenseLayer.initialize[A, D, D](initialization, hiddenActivation))
        .sequence

      output <- DenseLayer
        .initialize[A, O, D](initialization, outputActivation)
    yield WithHidden(firstHidden, additionalHidden, output)
