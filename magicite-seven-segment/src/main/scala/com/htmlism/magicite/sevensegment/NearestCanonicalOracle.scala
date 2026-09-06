package com.htmlism.magicite.sevensegment

/**
  * Future recognition-only reference based on nearest Hamming distance to canonical glyphs.
  *
  * It intentionally remains separate from `CorruptionRecovery`: it recognizes what was observed, while corruption
  * recovery asks which canonical source could have existed before a segment was damaged.
  */
object NearestCanonicalOracle
