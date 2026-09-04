package com.htmlism.magicite

import scala.util.Random

import weaver.*

object RngSpec extends FunSuite:
  test("shuffle returns a deterministic permutation from the same seed"):
    val xs =
      List("alpha", "bravo", "charlie", "delta", "echo")

    val actualA =
      Rng.shuffle(xs).runA(Random(123)).value

    val actualB =
      Rng.shuffle(xs).runA(Random(123)).value

    expect.same(actualA, actualB) &&
    expect.same(xs.toSet, actualA.toSet) &&
    expect.same(xs.length, actualA.length)

  test("shuffle can be composed as an Rng program"):
    val xs =
      List(1, 2, 3, 4)

    val expectedRng =
      Random(456)

    val expectedFirst =
      Rng.shuffle(xs).runA(expectedRng).value

    val expectedSecond =
      Rng.shuffle(xs).runA(expectedRng).value

    val program =
      for
        first  <- Rng.shuffle(xs)
        second <- Rng.shuffle(xs)
      yield first -> second

    val (first, second) =
      program.runA(Random(456)).value

    expect.same(expectedFirst, first) &&
    expect.same(expectedSecond, second) &&
    expect.same(xs.toSet, first.toSet) &&
    expect.same(xs.toSet, second.toSet)
