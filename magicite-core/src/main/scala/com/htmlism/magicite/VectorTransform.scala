package com.htmlism.magicite

/**
  * Used to transform the output of a vector of neurons
  */
enum VectorTransform:
  /** Leaves every vector value unchanged */
  case NoOp

  /** Converts vector scores into a probability distribution */
  case Softmax
