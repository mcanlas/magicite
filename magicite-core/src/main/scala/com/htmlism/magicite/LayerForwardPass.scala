package com.htmlism.magicite

/**
  * Values produced by one dense layer during a forward pass
  *
  * @tparam A
  *   The scalar type of the layer values
  * @tparam M
  *   The output dimension, with one coordinate for each output neuron
  *
  * @param preActivations
  *   Affine output values before the layer activation
  * @param outputs
  *   Values after the layer activation
  */
final case class LayerForwardPass[A, M](
    preActivations: Vec[A, M],
    outputs: Vec[A, M]
)
