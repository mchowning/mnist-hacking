import breeze.linalg.{DenseMatrix, DenseVector}

// TODO use Streams
// TODO use 3d matrix instead of list of 2d matrices for weights

/**
  * @param userWeights each DenseMatrix represents the weights going from a particular node in
  *                    layer l (row #) to a particular node in layer l+1 (column #)
  */
// TODO fix this constructor, don't need both numNodesPerLayer and userWeights
class NeuralNet(numNodesPerLayer: List[Int],
                userWeights: List[DenseMatrix[Double]] = Nil,
                activationFunc: ActivationFunc = NeuralNet.DefaultActivationFunc,
                costFunc: CostFunc = NeuralNet.DefaultCostFunc) {

  val weights = if (userWeights != Nil) userWeights else randomWeights()

  def randomWeights() = {
    println("Initializing with random weights...")
    // List of tuples where each tuple represents (# input nodes, # output nodes) from one layer to the next
    // adding one node to first layer to get weights for bias node (not adding to layer2 because bias nodes take no input)
    numNodesPerLayer.zip(numNodesPerLayer.tail)
      .map { t => DenseMatrix.rand(t._1 + 1, t._2) }
//      .map { t => DenseMatrix.rand(t._1, t._2, new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0))).uniform) }
  }

  def totalError(inputs: List[Double], expected: DenseVector[Double]): Double = {
    val output = forwardPropagate(inputs)
    costFunc(expected, output)
  }

  def train(inputs: List[Double], expected: DenseVector[Double]) = {
    val output: DenseVector[Double] = forwardPropagate(inputs)
    val cost = costFunc(expected, output)
    // TODO backprop
  }

  def forwardPropagate(inputs: List[Double]): DenseVector[Double] = {
    val inputVals = DenseVector(inputs.toArray)
    weights.foldLeft(inputVals) { (iVals, wVals) =>
      // TODO find better way to add to a vector
      val inputValsWithBias = DenseVector(iVals.toArray :+ 1d)
      (wVals.t * inputValsWithBias)  // multiply weights and inputs
        .map(activationFunc.apply)   // run results through activation function
    }
  }
}

object NeuralNet {
  val DefaultActivationFunc: ActivationFunc = Sigmoid
  val DefaultCostFunc: CostFunc = ResidualSumOfSquares
}
