package com.htmlism.magicite

import weaver.*

object MagiciteSuite extends FunSuite:
  test("it has a name"):
    expect.eql("magicite", Magicite.name)
