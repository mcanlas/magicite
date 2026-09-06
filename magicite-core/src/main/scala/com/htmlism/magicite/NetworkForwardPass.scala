package com.htmlism.magicite

/**
  * Values produced by a whole network during one forward pass
  *
  * @tparam A
  *   The scalar type of the network values
  * @tparam I
  *   The input dimension tag
  * @tparam O
  *   The output dimension tag
  */
sealed trait NetworkForwardPass[A, I, O]:
  /** The final network output for this pass */
  def prediction: Vec[A, O]

object NetworkForwardPass:
  /**
    * Forward-pass values for a direct network
    *
    * @tparam A
    *   The scalar type of the network values
    * @tparam I
    *   The input dimension tag
    * @tparam O
    *   The output dimension tag
    *
    * @param input
    *   The vector supplied to the output layer
    * @param outputPass
    *   The output layer's forward-pass values
    */
  final case class Direct[A, I, O](
      input: Vec[A, I],
      outputPass: LayerForwardPass[A, I, O]
  ) extends NetworkForwardPass[A, I, O]:
    def prediction: Vec[A, O] =
      outputPass.outputs

  /**
    * Forward-pass values for a network with hidden layers
    *
    * @tparam A
    *   The scalar type of the network values
    * @tparam I
    *   The input dimension tag
    * @tparam D
    *   The shared hidden-layer dimension tag
    * @tparam O
    *   The output dimension tag
    *
    * @param input
    *   The vector supplied to the first hidden layer
    * @param firstHidden
    *   The first hidden layer's forward-pass values
    * @param additionalHidden
    *   Forward-pass values for additional `D → D` hidden layers
    * @param outputPass
    *   The output layer's forward-pass values
    */
  final case class WithHidden[A, I, D, O](
      input: Vec[A, I],
      firstHidden: LayerForwardPass[A, I, D],
      additionalHidden: Vector[LayerForwardPass[A, D, D]],
      outputPass: LayerForwardPass[A, D, O]
  ) extends NetworkForwardPass[A, I, O]:
    def prediction: Vec[A, O] =
      outputPass.outputs
