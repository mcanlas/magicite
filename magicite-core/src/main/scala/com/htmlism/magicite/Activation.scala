package com.htmlism.magicite

enum Activation:
  case Identity
  case Relu
  case Tanh
  case Sigmoid

  def apply(x: Double): Double =
    this match
      case Identity => x
      case Relu     => Math.max(0.0, x)
      case Tanh     => Math.tanh(x)
      case Sigmoid  => 1.0 / (1.0 + Math.exp(-x))
