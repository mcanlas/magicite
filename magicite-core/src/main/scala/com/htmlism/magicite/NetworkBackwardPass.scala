package com.htmlism.magicite

/**
  * Gradients produced by a whole network during backpropagation
  *
  * @tparam A
  *   The scalar type of the network values and gradients
  * @tparam I
  *   The input dimension tag
  * @tparam O
  *   The output dimension tag
  */
sealed trait NetworkBackwardPass[A, I, O]

object NetworkBackwardPass:
  /**
    * Backward-pass values for a direct network
    *
    * @tparam A
    *   The scalar type of the network values and gradients
    * @tparam I
    *   The input dimension tag
    * @tparam O
    *   The output dimension tag
    *
    * @param outputPass
    *   The gradients produced by the direct output layer
    */
  final case class Direct[A, I, O](
      outputPass: LayerBackwardPass[A, I, O]
  ) extends NetworkBackwardPass[A, I, O]

  /**
    * Backward-pass values for a network with hidden layers
    *
    * @tparam A
    *   The scalar type of the network values and gradients
    * @tparam I
    *   The input dimension tag
    * @tparam D
    *   The shared hidden-layer dimension tag
    * @tparam O
    *   The output dimension tag
    *
    * @param firstHidden
    *   The gradients produced by the first hidden layer
    * @param additionalHidden
    *   Gradients produced by additional `D → D` hidden layers in forward order
    * @param outputPass
    *   The gradients produced by the output layer
    */
  final case class WithHidden[A, I, D, O](
      firstHidden: LayerBackwardPass[A, I, D],
      additionalHidden: Vector[LayerBackwardPass[A, D, D]],
      outputPass: LayerBackwardPass[A, D, O]
  ) extends NetworkBackwardPass[A, I, O]
