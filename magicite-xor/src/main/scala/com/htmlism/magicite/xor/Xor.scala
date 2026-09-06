package com.htmlism.magicite.xor

import scala.util.Random

import com.htmlism.magicite.*

/** A deterministic XOR experiment built from Magicite's core network primitives */
object Xor:
  /** The two-feature dimension of an XOR truth-table row */
  sealed trait XorOperands

  /** The four-value dimension tag used by the experiment's hidden layer */
  sealed trait D4

  /** The one-probability dimension of a Boolean result */
  sealed trait BooleanOutput

  given Dimension[XorOperands]   = Dimension(2)
  given Dimension[D4]            = Dimension(4)
  given Dimension[BooleanOutput] = Dimension(1)

  /**
    * The XOR experiment's `XorOperands → D4 → BooleanOutput` model shape
    *
    * @tparam A
    *   The real-valued scalar type used by the model parameters and values
    */
  type Model[A] =
    Network.WithHidden[A, XorOperands, D4, BooleanOutput]

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
    * The result of training an XOR network for whole epochs
    *
    * @tparam A
    *   The real-valued scalar type used by the model parameters, losses, and predictions
    *
    * @param network
    *   The network after its final parameter update
    * @param epochLosses
    *   Mean binary cross-entropy for each epoch before that epoch's updates
    */
  final case class TrainingResult[A](
      network: Model[A],
      epochLosses: Vector[A]
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
    * @tparam A
    *   The real-valued scalar type used by the encoded features
    *
    * @param left
    *   The left Boolean input
    * @param right
    *   The right Boolean input
    */
  def encodeInputs[A: RealScalar as scalar](left: Boolean, right: Boolean): Vec[A, XorOperands] =
    Vec[A, XorOperands](scalar.tabulate(2):
      case 0 => encodeBoolean(left)
      case _ => encodeBoolean(right))

  /**
    * Encodes a Boolean XOR result as the binary cross-entropy target
    *
    * @tparam A
    *   The real-valued scalar type used by the encoded target
    *
    * @param expected
    *   The expected Boolean XOR result
    */
  def encodeTarget[A: RealScalar](expected: Boolean): A =
    encodeBoolean(expected)

  /**
    * Decodes a sigmoid probability as a Boolean prediction using a one-half threshold
    *
    * @tparam A
    *   The real-valued scalar type used by the sigmoid probability
    *
    * @param probability
    *   The sigmoid probability of a true Boolean result
    */
  def decodePrediction[A: RealScalar as scalar](probability: A): Boolean =
    !scalar.isPositive(scalar.fromDouble(0.5) - probability)

  /**
    * Initializes the experiment's `2 → 4 → 1` network from a reproducible seed
    *
    * @tparam A
    *   The real-valued scalar type used by the initialized parameters
    *
    * @param seed
    *   The seed used for Xavier weight draws
    */
  def initialize[A: RealScalar](seed: Long): Model[A] =
    Network
      .withHidden[A, XorOperands, D4, BooleanOutput](
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
    * @tparam A
    *   The real-valued scalar type used by the network, loss, and learning rate
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
  def trainRow[A: RealScalar as scalar](
      network: Model[A],
      row: TruthTableRow,
      learningRate: A
  ): (Model[A], A) =
    val forwardPass =
      network.forward(encodeInputs[A](row.left, row.right))

    val target =
      encodeTarget[A](row.expected)

    val prediction =
      forwardPass.prediction.values(0)

    val loss =
      BinaryCrossEntropy.value(target, prediction)

    val outputGradient =
      BinaryCrossEntropy.sigmoidPreActivationDerivative(target, prediction)

    val backwardPass =
      network.backwardFromOutputPreActivation(
        forwardPass,
        Vec[A, BooleanOutput](scalar.tabulate(1)(_ => outputGradient))
      )

    network.updated(backwardPass, learningRate) -> loss

  /**
    * Trains the network for fixed-order, one-row gradient-descent epochs
    *
    * @tparam A
    *   The real-valued scalar type used by the network, losses, and learning rate
    *
    * @param initialNetwork
    *   The network used before the first training update
    * @param epochCount
    *   The positive number of full truth-table passes to perform
    * @param learningRate
    *   The positive scale of each truth-table row's gradient-descent step
    */
  def train[A: RealScalar as scalar](
      initialNetwork: Model[A],
      epochCount: Int,
      learningRate: A
  ): TrainingResult[A] =
    require(epochCount > 0, s"epoch count must be positive, but was $epochCount")
    require(scalar.isPositive(learningRate), s"learning rate must be positive")

    val (network, epochLosses) =
      (0 until epochCount).foldLeft(initialNetwork -> Vector.empty[A]):
        case ((currentNetwork, losses), _) =>
          val (updatedNetwork, totalLoss) =
            truthTable.foldLeft(currentNetwork -> scalar.zero):
              case ((network, loss), row) =>
                val (updatedNetwork, rowLoss) =
                  trainRow(network, row, learningRate)

                updatedNetwork -> (loss + rowLoss)

          updatedNetwork -> (losses :+ totalLoss / scalar.fromDouble(truthTable.size.toDouble))

    TrainingResult(network, epochLosses)

  /**
    * Produces the sigmoid probability for two Boolean XOR inputs
    *
    * @tparam A
    *   The real-valued scalar type used by the network and probability
    *
    * @param network
    *   The trained or untrained XOR network to evaluate
    * @param left
    *   The left Boolean input
    * @param right
    *   The right Boolean input
    */
  def predictProbability[A: RealScalar](
      network: Model[A],
      left: Boolean,
      right: Boolean
  ): A =
    network.forward(encodeInputs[A](left, right)).prediction.values(0)

  /**
    * Produces a Boolean XOR prediction from two Boolean inputs
    *
    * @tparam A
    *   The real-valued scalar type used by the network
    *
    * @param network
    *   The trained or untrained XOR network to evaluate
    * @param left
    *   The left Boolean input
    * @param right
    *   The right Boolean input
    */
  def predict[A: RealScalar](
      network: Model[A],
      left: Boolean,
      right: Boolean
  ): Boolean =
    decodePrediction(predictProbability(network, left, right))

  private def encodeBoolean[A: RealScalar](value: Boolean): A =
    if value then 1.0.toScalar else 0.0.toScalar
