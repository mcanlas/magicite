lazy val root =
  Project("magicite", file("."))
    .aggregate(core, xor, sevenSegment, ticTacToe, connectThree, othello)

lazy val core =
  module("core")
    .settings(description := "Neural networks in Scala")
    .withCats
    .withEffectMonad
    .withTesting
    .enablePublishing

lazy val xor =
  module("xor")
    .dependsOn(core)
    .withTesting

lazy val sevenSegment =
  module("seven-segment")
    .dependsOn(core)
    .withTesting

lazy val ticTacToe =
  module("tic-tac-toe")
    .dependsOn(core)
    .withTesting

lazy val connectThree =
  module("connect-three")
    .dependsOn(core)
    .withTesting

lazy val othello =
  module("othello")
    .dependsOn(core)
    .withTesting
