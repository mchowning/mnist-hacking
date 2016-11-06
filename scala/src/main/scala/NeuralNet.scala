import breeze.linalg.{*, DenseMatrix, DenseVector}

// TODO use Streams
// TODO use 3d matrix instead of list of 2d matrices for weights?

/**
  * @param weights each DenseMatrix represents the weights going from a particular node in
  *                    layer l (row #) to a particular node in layer l+1 (column #)
  */
class NeuralNet private (weights:         List[DenseMatrix[Double]],
                         activationFunc:  ActivationFunc,
                         costFunc:        CostFunc,
                         learningRate:    Double) {

  checkThatWeightsAreValid()

  private def checkThatWeightsAreValid() = {
    weights zip weights.tail foreach { case (w1, w2) =>
      if (w1.cols != w2.rows - 1) throw new Exception("Invalid Weights: weight matrices must match up so that the output " +
        "nodes (cols) for weights on a level match the number of input nodes (rows) - 1 (for the bias) for weights on the next level.")
    }
  }

  def totalError(inputs: List[Double], expected: DenseVector[Double]): Double = {
    val output = forwardPropagate(inputs)
    costFunc(expected, output)
  }

  def train(inputs: List[Double], expected: DenseVector[Double]): List[DenseMatrix[Double]] = {
      val activations: List[DenseVector[Double]] = forwardPropActivations(inputs)
      backpropagate(expected, activations)
  }

  /**
    * @return the new weight values that result from the backpropagation
    */
  private def backpropagate(expected: DenseVector[Double],
                            activations: List[DenseVector[Double]]): List[DenseMatrix[Double]] = {
    val output = activations.last
    val dcda = costFunc.derivative(expected, output)

    // Adding matrix of identityWeights for "weights" between final node(s) and error calculationss
    val modifiedWeights = weights :+ DenseMatrix.eye[Double](weights.last.cols)

    val weightActivationPairs = modifiedWeights
      .reverse zip activations.reverse

    // TODO try scanRight without reversing?
    // derivative of total error with respect to pre-neuronal activation
    val nodeErrors = weightActivationPairs.scanLeft(dcda) {
      case (upperNodeError, (weightMatrix, activationsBelow)) =>
        val deda = weightMatrix * upperNodeError
        val dadz = activationsBelow.map(activationFunc.derivative)
        dadz :* deda
    }.tail // drop the initial (reversed) term which only relates to the cost function level

    val dedw = nodeErrors.init zip activations.init.reverse map {
      case (errors, acts) => errors :* acts
    }

    (weights.reverse zip dedw).map { case (ws, ds) =>
      ws(*, ::) - (ds * learningRate)
    }.reverse
  }

  private def forwardPropActivations(inputs: List[Double]): List[DenseVector[Double]] = {
    val inputVals = DenseVector(inputs.toArray)
    weights.scanLeft(inputVals) { (iVals, wVals) =>
      // TODO find better way to add to a vector
      val inputValsWithBias = DenseVector(iVals.toArray :+ 1d)
      (wVals.t * inputValsWithBias)  // multiply weights and inputs
        .map(activationFunc.apply)   // run results through activation function
    }
  }

  def forwardPropagate(inputs: List[Double]): DenseVector[Double] = {
    forwardPropActivations(inputs).last
  }
}

object NeuralNet {

  private val DefaultActivationFunc = Sigmoid
  private val DefaultCostFunc = ResidualSumOfSquares
  private val DefaultLearningRate: Double = 0.5

  def withLayers(numNodesPerLayer:  List[Int],
                 activationFunc:    ActivationFunc  = DefaultActivationFunc,
                 costFunc:          CostFunc        = DefaultCostFunc,
                 learningRate:      Double          = DefaultLearningRate) = {

    println("Initializing with random weights...")
    // Adding 1 to each 'input' layer to get weights for bias node (not adding to 'output' layer because
    // bias nodes take no input)
    val randomWeights = (numNodesPerLayer zip numNodesPerLayer.tail) map {
      t => DenseMatrix.rand[Double](t._1 + 1, t._2)
      // t => DenseMatrix.rand[Double](t._1, t._2, new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0))).uniform)
    }

    withWeights(randomWeights, activationFunc, costFunc, learningRate)
  }

  def withWeights(weights:        List[DenseMatrix[Double]],
                  activationFunc: ActivationFunc  = DefaultActivationFunc,
                  costFunc:       CostFunc        = DefaultCostFunc,
                  learningRate:   Double          = DefaultLearningRate) = {
    new NeuralNet(weights, activationFunc, costFunc, learningRate)
  }
}
