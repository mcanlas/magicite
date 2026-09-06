package com.htmlism.magicite

import cats.Show
import org.scalacheck.Gen
import weaver.*
import weaver.scalacheck.Checkers

object VecProperties extends SimpleIOSuite with Checkers:
  import TestDimensions.*
  import TestDimensions.given

  private given Show[Vec[Double, Three]] =
    Show.show(vector => vector.values.mkString("Vec(", ", ", ")"))

  private val vectorOfThree =
    Gen
      .listOfN(3, Gen.choose(-100.0, 100.0))
      .map: values =>
        Vec[Double, Three](values.toArray)

  private val vectorPair: Gen[(Vec[Double, Three], Vec[Double, Three])] =
    for
      left  <- vectorOfThree
      right <- vectorOfThree
    yield (left, right)

  test("addition is commutative for equal-sized vectors"):
    forall(vectorPair): (left, right) =>
      expect.eql((left + right).values.toVector, (right + left).values.toVector)

  test("dot products are symmetric for equal-sized vectors"):
    forall(vectorPair): (left, right) =>
      expect.eql(left.dot(right), right.dot(left))
