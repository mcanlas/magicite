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

  /**
    * Computes this activation's derivative for one neuron
    *
    * @tparam A
    *   The real-valued scalar type of the cached values and derivative
    * @param preActivation
    *   The cached affine output before this activation
    * @param output
    *   The cached result after this activation
    */
  def derivative[A: RealScalar as scalar](preActivation: A, output: A): A =
    this match
      case Identity => scalar.one
      case Relu     => if scalar.isPositive(preActivation) then scalar.one else scalar.zero
      case Tanh     => scalar.one - output * output
      case Sigmoid  => output * (scalar.one - output)
