package com.htmlism.magicite

import weaver.*

object VectorTransformSuite extends FunSuite:
  test("has exactly the supported transforms"):
    expect.eql(
      Set(VectorTransform.NoOp, VectorTransform.Softmax),
      VectorTransform.values.toSet
    )
