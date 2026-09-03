lazy val root =
  Project("magicite", file("."))
    .aggregate(core)

lazy val core =
  module("core")
    .settings(description := "Neural networks in Scala")
    .withCats
    .withEffectMonad
    .withTesting
    .enablePublishing
