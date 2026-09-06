package com.htmlism.magicite.sevensegment

/** The seven binary segment values of one display, in `SevenSegment.Segment` order */
opaque type SegmentState = Vector[Int]

object SegmentState:
  /** Creates a valid seven-segment display state */
  def apply(values: Int*): SegmentState =
    fromVector(values.toVector)

  /** Creates a valid seven-segment display state from its ordered binary values */
  def fromVector(values: Vector[Int]): SegmentState =
    SevenSegment.requireSegments(values)

    values

  extension (state: SegmentState)
    /** The ordered binary values for interoperability at the collection boundary */
    def toVector: Vector[Int] = state

    /** The value of one named segment */
    def apply(segment: SevenSegment.Segment): Int =
      state(segment.ordinal)

    /** Returns a state with one segment replaced by a binary value */
    def updated(segment: SevenSegment.Segment, value: Int): SegmentState =
      fromVector(state.updated(segment.ordinal, value))
