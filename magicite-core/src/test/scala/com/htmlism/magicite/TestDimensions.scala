package com.htmlism.magicite

object TestDimensions:
  sealed trait One
  sealed trait Two
  sealed trait Three

  given Dimension[One]   = Dimension(1)
  given Dimension[Two]   = Dimension(2)
  given Dimension[Three] = Dimension(3)
