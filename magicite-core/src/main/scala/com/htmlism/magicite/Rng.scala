package com.htmlism.magicite

import scala.util.Random

import cats.data.State

type Rng[A] = State[Random, A]

object Rng:
  def nextInt(bound: Int): Rng[Int] =
    State: rng =>
      rng -> rng.nextInt(bound)

  def nextDouble: Rng[Double] =
    State: rng =>
      rng -> rng.nextDouble()

  def shuffle[A](xs: List[A]): Rng[List[A]] =
    State: rng =>
      rng -> rng.shuffle(xs)
