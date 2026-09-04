package com.htmlism.magicite

/**
  * A layer of neurons that is fully connected to the previous layer
  *
  * @tparam M
  *   The output dimension, with one coordinate for each output neuron
  * @tparam N
  *   The input dimension, with one coordinate for each input feature
  */
final case class DenseLayer[M, N](
    weights: Matrix[M, N],
    biases: Vec[M],
    activation: Activation
):
  require(
    biases.size == weights.rows,
    s"layer has ${weights.rows} output rows but ${biases.size} biases"
  )
