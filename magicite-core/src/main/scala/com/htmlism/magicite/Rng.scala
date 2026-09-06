package com.htmlism.magicite

import scala.util.Random

import cats.data.State

/**
  * A state program that consumes a seeded random generator and produces `A`
  *
  * @tparam A
  *   The value produced by the random program
  */
type Rng[A] = State[Random, A]

object Rng:
  def nextInt(bound: Int): Rng[Int] =
    State: rng =>
      rng -> rng.nextInt(bound)

  def nextDouble: Rng[Double] =
    State: rng =>
      rng -> rng.nextDouble()

  /**
    * Shuffles values using the program's random-generator state
    *
    * @tparam A
    *   The type of values being shuffled
    */
  def shuffle[A](xs: List[A]): Rng[List[A]] =
    State: rng =>
      rng -> rng.shuffle(xs)
