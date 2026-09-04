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
