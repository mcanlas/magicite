package com.htmlism.magicite

import weaver.*

object ActivationSuite extends FunSuite:
  test("identity leaves values unchanged"):
    expect.eql(-2.5, Activation.Identity(-2.5))

  test("relu removes negative values"):
    expect.all(
      Activation.Relu(-2.5) == 0.0,
      Activation.Relu(0.0) == 0.0,
      Activation.Relu(2.5) == 2.5
    )

  test("tanh is zero at zero and bounded"):
    expect.all(
      Activation.Tanh(0.0) == 0.0,
      Activation.Tanh(-10.0) > -1.0,
      Activation.Tanh(10.0) < 1.0
    )

  test("sigmoid is one half at zero and bounded"):
    expect.all(
      Activation.Sigmoid(0.0) == 0.5,
      Activation.Sigmoid(-10.0) > 0.0,
      Activation.Sigmoid(10.0) < 1.0
    )

  test("activations support Float scalars"):
    expect.all(
      Activation.Relu(-2.5f) == 0.0f,
      Activation.Tanh(0.0f) == 0.0f,
      Activation.Sigmoid(0.0f) == 0.5f
    )

  test("derivatives use cached pre-activation and output values"):
    expect.all(
      Activation.Identity.derivative(100.0, -100.0) == 1.0,
      Activation.Relu.derivative(-1.0, 0.0) == 0.0,
      Activation.Relu.derivative(0.0, 0.0) == 0.0,
      Activation.Relu.derivative(1.0, 1.0) == 1.0,
      Activation.Tanh.derivative(0.0, 0.5) == 0.75,
      Activation.Sigmoid.derivative(0.0, 0.25) == 0.1875
    )

  test("activation derivatives support Float scalars"):
    expect.eql(0.25f, Activation.Tanh.derivative(0.0f, 0.8660254f))
