package com.htmlism.magicite.xor

import scala.util.Random

import com.htmlism.magicite.*

/** A deterministic XOR experiment built from Magicite's core network primitives */
object Xor:
  /** The two-feature dimension of an XOR truth-table row */
  sealed trait XorInputs

  /** The two-value dimension tag used by the experiment's hidden layer */
  sealed trait D2

  /** The one-probability dimension of a Boolean result */
  sealed trait BooleanOutput

  given Dimension[XorInputs]     = Dimension(2)
  given Dimension[D2]            = Dimension(2)
  given Dimension[BooleanOutput] = Dimension(1)

  /**
    * One row of XOR's canonical truth table
    *
    * @param left
    *   The left Boolean input
    * @param right
    *   The right Boolean input
    * @param expected
    *   The expected Boolean XOR result
    */
  final case class TruthTableRow(
      left: Boolean,
      right: Boolean,
      expected: Boolean
  )

  /**
    * The result of training the XOR network for whole epochs
    *
    * @param network
    *   The network after its final parameter update
    * @param epochLosses
    *   Mean binary cross-entropy for each epoch before that epoch's updates
    */
  final case class TrainingResult(
      network: Network.WithHidden[Double, XorInputs, D2, BooleanOutput],
      epochLosses: Vector[Double]
  )

  /** The four XOR truth-table rows in a fixed training order */
  val truthTable: Vector[TruthTableRow] =
    Vector(
      TruthTableRow(left = false, right = false, expected = false),
      TruthTableRow(left = false, right = true, expected  = true),
      TruthTableRow(left = true, right  = false, expected = true),
      TruthTableRow(left = true, right  = true, expected  = false)
    )

  /**
    * Encodes two Boolean XOR inputs as the network's two numeric features
    *
    * @param left
    *   The left Boolean input
    * @param right
    *   The right Boolean input
    */
  def encodeInputs(left: Boolean, right: Boolean): Vec[Double, XorInputs] =
    Vec[Double, XorInputs](Array(encodeBoolean(left), encodeBoolean(right)))

  /**
    * Encodes a Boolean XOR result as the binary cross-entropy target
    *
    * @param expected
    *   The expected Boolean XOR result
    */
  def encodeTarget(expected: Boolean): Double =
    encodeBoolean(expected)

  /**
    * Decodes a sigmoid probability as a Boolean prediction using a one-half threshold
    *
    * @param probability
    *   The sigmoid probability of a true Boolean result
    */
  def decodePrediction(probability: Double): Boolean =
    probability >= 0.5

  /**
    * Initializes the experiment's `2 → 2 → 1` network from a reproducible seed
    *
    * @param seed
    *   The seed used for Xavier weight draws
    */
  def initialize(seed: Long): Network.WithHidden[Double, XorInputs, D2, BooleanOutput] =
    Network
      .withHidden[Double, XorInputs, D2, BooleanOutput](
        initialization   = Initialization.Xavier,
        hiddenLayerCount = 1,
        hiddenActivation = Activation.Tanh,
        outputActivation = Activation.Sigmoid
      )
      .runA(Random(seed))
      .value

  /**
    * Runs one stochastic-gradient update for one XOR truth-table row
    *
    * @param network
    *   The network before this row's update
    * @param row
    *   The truth-table input and target used for this update
    * @param learningRate
    *   The positive scale of the gradient-descent step
    * @return
    *   The updated network and this row's binary cross-entropy loss
    */
  def trainRow(
      network: Network.WithHidden[Double, XorInputs, D2, BooleanOutput],
      row: TruthTableRow,
      learningRate: Double
  ): (Network.WithHidden[Double, XorInputs, D2, BooleanOutput], Double) =
    val forwardPass =
      network.forward(encodeInputs(row.left, row.right))

    val target =
      encodeTarget(row.expected)

    val prediction =
      forwardPass.prediction.values(0)

    val loss =
      BinaryCrossEntropy.value(target, prediction)

    val outputGradient =
      BinaryCrossEntropy.sigmoidPreActivationDerivative(target, prediction)

    val backwardPass =
      network.backwardFromOutputPreActivation(
        forwardPass,
        Vec[Double, BooleanOutput](Array(outputGradient))
      )

    network.updated(backwardPass, learningRate) -> loss

  /**
    * Trains the network for fixed-order, one-row gradient-descent epochs
    *
    * @param initialNetwork
    *   The network used before the first training update
    * @param epochCount
    *   The positive number of full truth-table passes to perform
    * @param learningRate
    *   The positive scale of each truth-table row's gradient-descent step
    */
  def train(
      initialNetwork: Network.WithHidden[Double, XorInputs, D2, BooleanOutput],
      epochCount: Int,
      learningRate: Double
  ): TrainingResult =
    require(epochCount > 0, s"epoch count must be positive, but was $epochCount")
    require(learningRate > 0.0, s"learning rate must be positive, but was $learningRate")

    val (network, epochLosses) =
      (0 until epochCount).foldLeft(initialNetwork -> Vector.empty[Double]):
        case ((currentNetwork, losses), _) =>
          val (updatedNetwork, totalLoss) =
            truthTable.foldLeft(currentNetwork -> 0.0):
              case ((network, loss), row) =>
                val (updatedNetwork, rowLoss) =
                  trainRow(network, row, learningRate)

                updatedNetwork -> (loss + rowLoss)

          updatedNetwork -> (losses :+ totalLoss / truthTable.size)

    TrainingResult(network, epochLosses)

  /**
    * Produces the sigmoid probability for two Boolean XOR inputs
    *
    * @param network
    *   The trained or untrained XOR network to evaluate
    * @param left
    *   The left Boolean input
    * @param right
    *   The right Boolean input
    */
  def predictProbability(
      network: Network.WithHidden[Double, XorInputs, D2, BooleanOutput],
      left: Boolean,
      right: Boolean
  ): Double =
    network.forward(encodeInputs(left, right)).prediction.values(0)

  /**
    * Produces a Boolean XOR prediction from two Boolean inputs
    *
    * @param network
    *   The trained or untrained XOR network to evaluate
    * @param left
    *   The left Boolean input
    * @param right
    *   The right Boolean input
    */
  def predict(
      network: Network.WithHidden[Double, XorInputs, D2, BooleanOutput],
      left: Boolean,
      right: Boolean
  ): Boolean =
    decodePrediction(predictProbability(network, left, right))

  private def encodeBoolean(value: Boolean): Double =
    if value then 1.0 else 0.0
