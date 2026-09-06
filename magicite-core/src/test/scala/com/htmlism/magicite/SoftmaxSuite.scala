package com.htmlism.magicite

import weaver.*

object SoftmaxSuite extends FunSuite:
  sealed trait ThreeClasses

  given Dimension[ThreeClasses] = Dimension(3)

  test("converts logits into probabilities that sum to one"):
    val probabilities =
      Softmax.probabilities(Vec[Double, ThreeClasses](Array(1.0, 2.0, 3.0)))

    expect.all(
      approximatelyEqual(probabilities.values(0), 0.09003057317038046),
      approximatelyEqual(probabilities.values(1), 0.24472847105479764),
      approximatelyEqual(probabilities.values(2), 0.6652409557748218),
      approximatelyEqual(probabilities.values.sum, 1.0)
    )

  test("is unchanged when every logit is shifted by the same value"):
    val original =
      Softmax.probabilities(Vec[Double, ThreeClasses](Array(-1.0, 0.0, 1.0)))

    val shifted =
      Softmax.probabilities(Vec[Double, ThreeClasses](Array(999.0, 1000.0, 1001.0)))

    val equivalent =
      original
        .values
        .zip(shifted.values)
        .forall:
          case (left, right) => approximatelyEqual(left, right)

    expect(equivalent)

  test("keeps large finite logits finite"):
    val probabilities =
      Softmax.probabilities(Vec[Double, ThreeClasses](Array(1000.0, 1001.0, 1002.0)))

    expect.all(
      probabilities.values.forall(_.isFinite),
      approximatelyEqual(probabilities.values.sum, 1.0)
    )

  test("supports Float scalars"):
    val probabilities =
      Softmax.probabilities(Vec[Float, ThreeClasses](Array(1.0f, 2.0f, 3.0f)))

    expect(math.abs(probabilities.values.sum - 1.0f) < 1e-6f)

  private def approximatelyEqual(left: Double, right: Double): Boolean =
    math.abs(left - right) < 1e-12
