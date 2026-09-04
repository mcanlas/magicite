package com.htmlism.magicite

/**
  * Used to transform the output of a vector of neurons
  */
enum VectorTransform:
  case NoOp
  case Softmax
