package com.htmlism.magicite

/**
  * Gradients produced by one dense layer during backpropagation
  *
  * @tparam A
  *   The scalar type of the layer values and gradients
  * @tparam N
  *   The input dimension, with one coordinate for each input feature
  * @tparam M
  *   The output dimension, with one coordinate for each output neuron
  *
  * @param preActivationGradient
  *   The loss gradient with respect to each affine output
  * @param weightGradients
  *   The loss gradient with respect to the layer's weight matrix
  * @param biasGradients
  *   The loss gradient with respect to the layer's bias vector
  * @param inputGradient
  *   The loss gradient to pass to the preceding layer's outputs
  */
final case class LayerBackwardPass[A, N, M](
    preActivationGradient: Vec[A, M],
    weightGradients: Matrix[A, M, N],
    biasGradients: Vec[A, M],
    inputGradient: Vec[A, N]
)
