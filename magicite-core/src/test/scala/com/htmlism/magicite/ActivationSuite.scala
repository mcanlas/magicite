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
