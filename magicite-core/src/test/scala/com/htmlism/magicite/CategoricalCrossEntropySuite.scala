package com.htmlism.magicite

import weaver.*

object CategoricalCrossEntropySuite extends FunSuite:
  sealed trait ThreeClasses

  given Dimension[ThreeClasses] = Dimension(3)

  test("selects the negative log probability of a one-hot target"):
    val target =
      Vec[Double, ThreeClasses](Array(0.0, 1.0, 0.0))

    val probabilities =
      Vec[Double, ThreeClasses](Array(0.1, 0.7, 0.2))

    expect(approximatelyEqual(CategoricalCrossEntropy.value(target, probabilities), -math.log(0.7)))

  test("computes the derivative with respect to each probability"):
    val target =
      Vec[Double, ThreeClasses](Array(0.0, 1.0, 0.0))

    val probabilities =
      Vec[Double, ThreeClasses](Array(0.1, 0.7, 0.2))

    val derivative =
      CategoricalCrossEntropy.predictionDerivative(target, probabilities)

    expect(derivative.values.toVector == Vector(0.0, -1.0 / 0.7, 0.0))

  test("combines categorical cross-entropy and softmax derivatives"):
    val target =
      Vec[Double, ThreeClasses](Array(0.0, 1.0, 0.0))

    val probabilities =
      Vec[Double, ThreeClasses](Array(0.1, 0.7, 0.2))

    val derivative =
      CategoricalCrossEntropy.softmaxPreActivationDerivative(target, probabilities)

    expect.all(
      approximatelyEqual(derivative.values(0), 0.1),
      approximatelyEqual(derivative.values(1), -0.3),
      approximatelyEqual(derivative.values(2), 0.2)
    )

  test("matches finite differences through softmax for every logit"):
    val logits =
      Vec[Double, ThreeClasses](Array(-0.5, 0.25, 1.0))

    val target =
      Vec[Double, ThreeClasses](Array(0.0, 1.0, 0.0))

    val probabilities =
      Softmax.probabilities(logits)

    val analytic =
      CategoricalCrossEntropy.softmaxPreActivationDerivative(target, probabilities)

    val epsilon =
      1e-5

    val numerical =
      logits
        .values
        .indices
        .map: i =>
          (loss(withLogit(logits, i, logits.values(i) + epsilon), target) -
            loss(withLogit(logits, i, logits.values(i) - epsilon), target)) / (2.0 * epsilon)

    val matches =
      analytic
        .values
        .zip(numerical)
        .forall:
          case (left, right) => approximatelyEqual(left, right)

    expect(matches)

  test("supports Float scalars"):
    val target =
      Vec[Float, ThreeClasses](Array(0.0f, 1.0f, 0.0f))

    val probabilities =
      Vec[Float, ThreeClasses](Array(0.1f, 0.7f, 0.2f))

    val derivative =
      CategoricalCrossEntropy.softmaxPreActivationDerivative(target, probabilities)

    expect(derivative.values.toVector == Vector(0.1f, -0.3f, 0.2f))

  private def loss(logits: Vec[Double, ThreeClasses], target: Vec[Double, ThreeClasses]): Double =
    CategoricalCrossEntropy.value(target, Softmax.probabilities(logits))

  private def withLogit(
      logits: Vec[Double, ThreeClasses],
      index: Int,
      value: Double
  ): Vec[Double, ThreeClasses] =
    val changed =
      logits.values.clone

    changed(index) = value

    Vec[Double, ThreeClasses](changed)

  private def approximatelyEqual(left: Double, right: Double): Boolean =
    math.abs(left - right) < 1e-8
