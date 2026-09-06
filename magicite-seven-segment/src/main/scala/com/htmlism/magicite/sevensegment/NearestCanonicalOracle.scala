package com.htmlism.magicite.sevensegment

/**
  * Recognition-only reference based on nearest Hamming distance to canonical glyphs.
  *
  * For any observed seven-segment pattern, this oracle returns the canonical digit(s) that differ in the fewest
  * segments. When a single canonical digit is closest it is the unique prediction; when multiple digits tie for the
  * smallest Hamming distance the pattern is ambiguous.
  *
  * It intentionally remains separate from `CorruptionRecovery`: it recognizes what was observed, while corruption
  * recovery asks which canonical source could have existed before a segment was damaged.
  */
object NearestCanonicalOracle:
  /** A non-negative count of differing segments between two seven-segment patterns */
  opaque type HammingDistance = Int

  object HammingDistance:
    /** Wraps a non-negative segment-difference count */
    def apply(value: Int): HammingDistance =
      require(value >= 0, s"Hamming distance must be non-negative, but was $value")
      value

    extension (hd: HammingDistance)
      /** The underlying non-negative integer count */
      def toInt: Int = hd

  /** The Hamming distance between two seven-segment patterns */
  def hammingDistance(left: SegmentState, right: SegmentState): HammingDistance =
    left
      .toVector
      .zip(right.toVector)
      .count: (a, b) =>
        a != b

  /** Every canonical digit together with its Hamming distance to the observed pattern, sorted by distance then digit */
  def distances(segments: SegmentState): Vector[(Int, HammingDistance)] =
    SevenSegment
      .canonicalDigits
      .map: canonical =>
        canonical.digit -> hammingDistance(segments, canonical.segments)
      .sortBy: (digit, distance) =>
        (distance, digit)

  /** The smallest Hamming distance between the observed pattern and any canonical digit */
  def minimumDistance(segments: SegmentState): HammingDistance =
    SevenSegment
      .canonicalDigits
      .map: canonical =>
        hammingDistance(segments, canonical.segments)
      .foldLeft(SevenSegment.Segment.values.length): (best, distance) =>
        Math.min(best, distance)

  /** Every canonical digit that achieves the minimum Hamming distance, sorted by digit */
  def nearest(segments: SegmentState): Vector[Int] =
    val minDist =
      minimumDistance(segments)

    SevenSegment
      .canonicalDigits
      .collect:
        case canonical if hammingDistance(segments, canonical.segments) == minDist =>
          canonical.digit

  /** Whether the observed pattern has exactly one nearest canonical digit */
  def isUnambiguous(segments: SegmentState): Boolean =
    nearest(segments).size == 1

  /**
    * The unique nearest canonical digit, or `None` when multiple digits tie.
    *
    * This is the deterministic "what digit does this look like?" answer that a recognition system would return when the
    * answer is clear-cut.
    */
  def predict(segments: SegmentState): Option[Int] =
    nearest(segments) match
      case Vector(single) => Some(single)
      case _              => None

  /**
    * The unique nearest canonical digit for a labelled display.
    *
    * Returns `None` when multiple canonical digits tie for the smallest Hamming distance.
    */
  def predict(row: SevenSegment.LabeledDisplay): Option[Int] =
    predict(row.segments)
