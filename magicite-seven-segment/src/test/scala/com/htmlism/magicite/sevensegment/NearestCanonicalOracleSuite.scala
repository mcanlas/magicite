package com.htmlism.magicite.sevensegment

import weaver.*

import com.htmlism.magicite.sevensegment.NearestCanonicalOracle.HammingDistance

/** Tests for the Hamming-distance recognition oracle */
object NearestCanonicalOracleSuite extends FunSuite:
  test("Hamming distance between identical patterns is zero"):
    val zero =
      SevenSegment.canonicalDigits(0).segments

    expect(NearestCanonicalOracle.hammingDistance(zero, zero).toInt == 0)

  test("Hamming distance between all-on and all-off is seven"):
    expect(
      NearestCanonicalOracle
        .hammingDistance(
          SegmentState(1, 1, 1, 1, 1, 1, 1),
          SegmentState(0, 0, 0, 0, 0, 0, 0)
        )
        .toInt == 7
    )

  test("Hamming distance is symmetric"):
    val a =
      SevenSegment.canonicalDigits(0).segments

    val b =
      SevenSegment.canonicalDigits(1).segments

    expect(
      NearestCanonicalOracle.hammingDistance(a, b).toInt ==
        NearestCanonicalOracle.hammingDistance(b, a).toInt
    )

  test("every canonical digit is its own nearest neighbor at distance zero"):
    val results =
      SevenSegment
        .canonicalDigits
        .map: canonical =>
          val nearestDigits =
            NearestCanonicalOracle.nearest(canonical.segments)

          val minDist =
            NearestCanonicalOracle.minimumDistance(canonical.segments)

          (canonical.digit, nearestDigits, minDist)

    expect.all(
      results.forall: (digit, nearestDigits, minDist) =>
        nearestDigits.contains(digit) && minDist.toInt == 0
    )

  test("every canonical digit is unambiguously predicted as itself"):
    val predictions =
      SevenSegment
        .canonicalDigits
        .map: canonical =>
          NearestCanonicalOracle.predict(canonical)

    val expectedDigits =
      SevenSegment
        .canonicalDigits
        .map: canonical =>
          Some(canonical.digit)

    expect(predictions == expectedDigits)

  test("a one-bit flip from digit 1 that forms digit 7 is recognized as digit 7"):
    // digit 1 = Vector(0, 1, 1, 0, 0, 0, 0), flip the top segment on
    val flipped =
      SegmentState(1, 1, 1, 0, 0, 0, 0)

    // that pattern is canonical digit 7; the oracle should return 7 since that's
    // the exact match (distance 0) rather than 1 (distance 1)
    expect(NearestCanonicalOracle.predict(flipped) == Some(7))

  test("distances returns all ten canonical digits sorted by distance then digit"):
    val ds =
      NearestCanonicalOracle.distances(SevenSegment.canonicalDigits(0).segments)

    expect.all(
      ds.size == 10,
      ds.headOption
        .exists: (digit, distance) =>
          digit == 0 && distance.toInt == 0,
      ds == ds.sortBy: (digit, distance) =>
        (distance.toInt, digit)
    )

  test("a pattern equidistant to multiple canonical digits is ambiguous"):
    val possibleSegmentPatterns =
      1 << SevenSegment.Segment.values.size

    val tiedPattern =
      (0 until possibleSegmentPatterns)
        .map: bits =>
          SevenSegment
            .Segment
            .values
            .indices
            .map: i =>
              // decode this integer into its seven on/off segment values
              (bits >> i) & 1
        .map: segments =>
          SegmentState.fromVector(segments.toVector)
        .find: segments =>
          NearestCanonicalOracle.nearest(segments).size > 1

    expect.all(
      tiedPattern.nonEmpty,
      tiedPattern.forall: segments =>
        !NearestCanonicalOracle.isUnambiguous(segments),
      tiedPattern.forall: segments =>
        NearestCanonicalOracle.predict(segments).isEmpty
    )

  test("every unambiguous one-bit corruption is predicted as a canonical digit"):
    val predictions =
      CorruptionRecovery
        .unambiguous
        .map: corruption =>
          NearestCanonicalOracle.predict(corruption.segments)

    // every unambiguous corruption should have a unique nearest canonical digit
    // (the distance is at most 1, so ties with other canonicals are unlikely for non-canonical patterns)
    expect(predictions.forall(_.isDefined))

  test("predict with LabeledDisplay delegates to segment-based predict"):
    val canonical =
      SevenSegment.canonicalDigits(5)

    expect(
      NearestCanonicalOracle.predict(canonical) ==
        NearestCanonicalOracle.predict(canonical.segments)
    )
