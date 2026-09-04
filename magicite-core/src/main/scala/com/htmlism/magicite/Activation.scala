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
    * Activations remain generic because [[Scalar]] explicitly includes the ordering and transcendental operations they
    * require.
    */
  def apply[A](x: A)(using scalar: Scalar[A]): A =
    this match
      case Identity => x
      case Relu     => scalar.maximum(scalar.zero, x)
      case Tanh     => scalar.tanh(x)
      case Sigmoid  => scalar.divide(scalar.one, scalar.add(scalar.one, scalar.exp(scalar.negate(x))))
