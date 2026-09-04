package com.htmlism.magicite

import cats.Show
import org.scalacheck.Gen
import weaver.*
import weaver.scalacheck.Checkers

object VectorNProperties extends SimpleIOSuite with Checkers:
  private given Show[VectorN] =
    Show.show(vector => vector.values.mkString("VectorN(", ", ", ")"))

  private def vectorOfSize(size: Int) =
    Gen
      .listOfN(size, Gen.choose(-100.0, 100.0))
      .map(values => VectorN(values.toArray))

  private val vectorPair: Gen[(VectorN, VectorN)] =
    for
      size  <- Gen.choose(0, 16)
      left  <- vectorOfSize(size)
      right <- vectorOfSize(size)
    yield (left, right)

  test("addition is commutative for equal-sized vectors"):
    forall(vectorPair): (left, right) =>
      expect.eql((left + right).values.toVector, (right + left).values.toVector)

  test("dot products are symmetric for equal-sized vectors"):
    forall(vectorPair): (left, right) =>
      expect.eql(left.dot(right), right.dot(left))
