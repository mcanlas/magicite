package com.htmlism.magicite

import cats.Show
import org.scalacheck.Gen
import weaver.*
import weaver.scalacheck.Checkers

object MatrixProperties extends SimpleIOSuite with Checkers:
  private given Show[VectorN] =
    Show.show(_.values.mkString("VectorN(", ", ", ")"))

  private given Show[Matrix] =
    Show.show(m => s"Matrix(${m.rows}, ${m.columns}, ${m.values.mkString("[", ", ", "]")})")

  private def matrixOfSize(rows: Int, columns: Int) =
    Gen
      .listOfN(rows * columns, Gen.choose(-10.0, 10.0))
      .map(values => Matrix(rows, columns, values.toArray))

  private def vectorOfSize(size: Int) =
    Gen
      .listOfN(size, Gen.choose(-10.0, 10.0))
      .map(values => VectorN(values.toArray))

  private def approximatelyEqual(left: Double, right: Double) =
    math.abs(left - right) <= 1e-10

  private val matrixAndVectorPairs: Gen[(Matrix, VectorN, VectorN)] =
    for
      rows    <- Gen.choose(0, 8)
      columns <- Gen.choose(0, 8)
      matrix  <- matrixOfSize(rows, columns)
      left    <- vectorOfSize(columns)
      right   <- vectorOfSize(columns)
    yield (matrix, left, right)

  private val dimensionsAndVector: Gen[(Int, Int, VectorN)] =
    for
      rows    <- Gen.choose(0, 8)
      columns <- Gen.choose(0, 8)
      input   <- vectorOfSize(columns)
    yield (rows, columns, input)

  test("multiplication distributes over vector addition"):
    forall(matrixAndVectorPairs): (matrix, left, right) =>
      val combined = matrix.multiply(left + right)
      val separate = matrix.multiply(left) + matrix.multiply(right)

      forEach(combined.values.iterator.zip(separate.values).toList): (actual, expected) =>
        expect(approximatelyEqual(actual, expected))

  test("a zero matrix maps every compatible vector to zero"):
    forall(dimensionsAndVector): (rows, columns, input) =>
      val output = Matrix(rows, columns, Array.fill(rows * columns)(0.0)).multiply(input)

      expect.eql(Vector.fill(rows)(0.0), output.values.toVector)
