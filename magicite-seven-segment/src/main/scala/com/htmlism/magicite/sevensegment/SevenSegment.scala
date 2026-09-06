package com.htmlism.magicite.sevensegment

import com.htmlism.magicite.*

/** The canonical seven-segment digit inputs used by the classification experiment */
object SevenSegment:
  /** The seven input features, in their fixed vector order */
  enum Segment:
    case Top, UpperRight, LowerRight, Bottom, LowerLeft, UpperLeft, Middle

  /** Phantom dimension for a seven-segment display's input vector */
  sealed trait Segments

  given Dimension[Segments] = Dimension(Segment.values.length)

  /** One labelled input row for digit classification */
  final case class TruthTableRow(digit: Int, segments: Vector[Int]):
    require(0 <= digit && digit <= 9, s"digit must be between 0 and 9, but was $digit")
    require(
      segments.length == Segment.values.length,
      s"a display must have ${Segment.values.length} segments, but had ${segments.length}"
    )
    require(segments.forall(bit => bit == 0 || bit == 1), s"segments must be binary, but were $segments")

  /**
    * The ten clean digit inputs in `[top, upper-right, lower-right, bottom, lower-left, upper-left, middle]` order
    */
  val truthTable: Vector[TruthTableRow] =
    Vector(
      TruthTableRow(0, Vector(1, 1, 1, 1, 1, 1, 0)),
      TruthTableRow(1, Vector(0, 1, 1, 0, 0, 0, 0)),
      TruthTableRow(2, Vector(1, 1, 0, 1, 1, 0, 1)),
      TruthTableRow(3, Vector(1, 1, 1, 1, 0, 0, 1)),
      TruthTableRow(4, Vector(0, 1, 1, 0, 0, 1, 1)),
      TruthTableRow(5, Vector(1, 0, 1, 1, 0, 1, 1)),
      TruthTableRow(6, Vector(1, 0, 1, 1, 1, 1, 1)),
      TruthTableRow(7, Vector(1, 1, 1, 0, 0, 0, 0)),
      TruthTableRow(8, Vector(1, 1, 1, 1, 1, 1, 1)),
      TruthTableRow(9, Vector(1, 1, 1, 1, 0, 1, 1))
    )

  /** Converts a canonical binary segment row into a numeric network input vector */
  def encodeInputs[A: RealScalar as scalar](row: TruthTableRow): Vec[A, Segments] =
    Vec[A, Segments](scalar.tabulate(Segment.values.length): index =>
      scalar.fromDouble(row.segments(index).toDouble))
