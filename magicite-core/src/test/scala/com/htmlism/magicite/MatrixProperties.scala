package com.htmlism.magicite

import cats.Show
import org.scalacheck.Gen
import weaver.*
import weaver.scalacheck.Checkers

import com.htmlism.magicite.Approx.approximatelyEqual

object MatrixProperties extends SimpleIOSuite with Checkers:
  import TestDimensions.*
  import TestDimensions.given

  private given Show[Vec[Double, Three]] =
    Show.show(_.values.mkString("Vec(", ", ", ")"))

  private given Show[Matrix[Double, Two, Three]] =
    Show.show(m => s"Matrix(${m.rows}, ${m.columns}, ${m.values.mkString("[", ", ", "]")})")

  private val matrixOfTwoByThree =
    Gen
      .listOfN(6, Gen.choose(-10.0, 10.0))
      .map: values =>
        Matrix[Double, Two, Three](values.toArray)

  private val vectorOfThree =
    Gen
      .listOfN(3, Gen.choose(-10.0, 10.0))
      .map: values =>
        Vec[Double, Three](values.toArray)

  private val matrixAndVectorPairs: Gen[(Matrix[Double, Two, Three], Vec[Double, Three], Vec[Double, Three])] =
    for
      matrix <- matrixOfTwoByThree
      left   <- vectorOfThree
      right  <- vectorOfThree
    yield (matrix, left, right)

  test("multiplication distributes over vector addition"):
    forall(matrixAndVectorPairs): (matrix, left, right) =>
      val combined = matrix.multiply(left + right)
      val separate = matrix.multiply(left) + matrix.multiply(right)

      forEach(combined.values.iterator.zip(separate.values).toList): (actual, expected) =>
        expect(approximatelyEqual(actual, expected, tolerance = 1e-10))

  test("a zero matrix maps every compatible vector to zero"):
    forall(vectorOfThree): input =>
      val output = Matrix[Double, Two, Three](Array.fill(6)(0.0)).multiply(input)

      expect.eql(Vector.fill(2)(0.0), output.values.toVector)
