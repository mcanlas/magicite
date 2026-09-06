package com.htmlism.magicite.sevensegment

/**
  * A test-scope console renderer for inspecting clean or corrupted seven-segment inputs.
  *
  * Run with `sbt 'seven-segment/Test/runMain com.htmlism.magicite.sevensegment.SevenSegmentDisplayApp'`. Replace
  * `inputs` in `main` with any `Array[Array[Int]]` of seven binary values to inspect other glyphs.
  */
object SevenSegmentDisplayApp:
  def main(args: Array[String]): Unit =
    val inputs =
      SevenSegment
        .canonicalDigits
        .map:
          _.segments.toVector.toArray
        .toArray

    printDisplays(inputs)

  /**
    * Prints one ASCII display for each `[top, upper-right, lower-right, bottom, lower-left, upper-left, middle]` input
    */
  def printDisplays(inputs: Array[Array[Int]]): Unit =
    inputs.foreach: input =>
      println(s"${input.mkString("[", ", ", "]")}")
      println(render(input))
      println()

  /** Renders one seven-segment input as five lines of ASCII art */
  def render(input: Array[Int]): String =
    require(input.length == SevenSegment.Segment.values.length, s"expected seven segments, but got ${input.length}")
    require(input.forall(bit => bit == 0 || bit == 1), s"segments must be binary, but were ${input.toSeq}")

    val top        = horizontal(input(SevenSegment.Segment.Top.ordinal))
    val upperRight = vertical(input(SevenSegment.Segment.UpperRight.ordinal))
    val lowerRight = vertical(input(SevenSegment.Segment.LowerRight.ordinal))
    val bottom     = horizontal(input(SevenSegment.Segment.Bottom.ordinal))
    val lowerLeft  = vertical(input(SevenSegment.Segment.LowerLeft.ordinal))
    val upperLeft  = vertical(input(SevenSegment.Segment.UpperLeft.ordinal))
    val middle     = horizontal(input(SevenSegment.Segment.Middle.ordinal))

    Vector(
      s" $top ",
      s"$upperLeft   $upperRight",
      s" $middle ",
      s"$lowerLeft   $lowerRight",
      s" $bottom "
    ).mkString("\n")

  private def horizontal(bit: Int): String =
    if bit == 1 then "---" else "   "

  private def vertical(bit: Int): String =
    if bit == 1 then "|" else " "
