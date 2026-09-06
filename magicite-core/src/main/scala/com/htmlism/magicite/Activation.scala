package com.htmlism.magicite

/**
  * Used to transform the output of a single neuron
  */
enum Activation:
  /** Returns its input unchanged */
  case Identity

  /** Returns zero for negative inputs and the input otherwise */
  case Relu

  /** Smoothly maps values into the interval from negative one to one */
  case Tanh

  /** Smoothly maps values into the interval from zero to one */
  case Sigmoid

  /**
    * Activations remain generic over real-valued scalar types.
    *
    * @tparam A
    *   The real-valued scalar type of the activation input and result
    */
  def apply[A: RealScalar as scalar](x: A): A =
    this match
      case Identity => x
      case Relu     => scalar.maximum(scalar.zero, x)
      case Tanh     => scalar.tanh(x)
      case Sigmoid  => scalar.one / (scalar.one + scalar.exp(-x))
