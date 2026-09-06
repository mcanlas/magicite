package com.htmlism.magicite

/**
  * Used to transform the output of a single neuron
  */
enum Activation:
  case Identity
  case Relu
  case Tanh
  case Sigmoid

  /**
    * Activations remain generic over real-valued scalar types.
    *
    * @tparam A
    *   The real-valued scalar type of the activation input and result
    */
  def apply[A](x: A)(using scalar: RealScalar[A]): A =
    this match
      case Identity => x
      case Relu     => scalar.maximum(scalar.zero, x)
      case Tanh     => scalar.tanh(x)
      case Sigmoid  => scalar.one / (scalar.one + scalar.exp(-x))
