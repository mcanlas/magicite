package com.htmlism.magicite

/**
  * A layer of neurons that is fully connected to the previous layer
  */
final case class DenseLayer(
    weights: Matrix,
    biases: VectorN,
    activation: Activation
):
  require(
    biases.size == weights.rows,
    s"layer has ${weights.rows} output rows but ${biases.size} biases"
  )
