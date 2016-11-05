import breeze.linalg.{*, DenseMatrix, DenseVector}

// TODO use Streams
// TODO use 3d matrix instead of list of 2d matrices for weights?

/**
  * @param userWeights each DenseMatrix represents the weights going from a particular node in
  *                    layer l (row #) to a particular node in layer l+1 (column #)
  */
// TODO fix this constructor, don't need both numNodesPerLayer and userWeights
class NeuralNet(numNodesPerLayer: List[Int],
                userWeights: List[DenseMatrix[Double]] = Nil,
                activationFunc: ActivationFunc = NeuralNet.DefaultActivationFunc,
                costFunc: CostFunc = NeuralNet.DefaultCostFunc) {

  val weights: List[DenseMatrix[Double]] = if (userWeights != Nil) userWeights else randomWeights()

  def randomWeights() = {
    println("Initializing with random weights...")
    // List of tuples where each tuple represents (# input nodes, # output nodes) from one layer to the next
    // adding one node to first layer to get weights for bias node (not adding to layer2 because bias nodes take no input)
    numNodesPerLayer.zip(numNodesPerLayer.tail)
      .map { t => DenseMatrix.rand(t._1 + 1, t._2): DenseMatrix[Double] }
//      .map { t => DenseMatrix.rand(t._1, t._2, new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0))).uniform) }
  }

  def totalError(inputs: List[Double], expected: DenseVector[Double]): Double = {
    val output = forwardPropagate(inputs)
    costFunc(expected, output)
  }

  def train(inputs: List[Double], expected: DenseVector[Double]) = {

    val activations: List[DenseVector[Double]] = forwardPropActivations(inputs)
    println(s"activations: $activations")
    val output: DenseVector[Double] = activations.last
    val cost = costFunc(expected, output)

    val dcda: DenseVector[Double] = costFunc.derivative(expected, output)
    println(s"dcda: $dcda")
//    val dadz = output.map(activationFunc.derivative)
//    println(s"dadz: $dadz")

    // Adding matrix of identityWeights for "weights" between final node(s) and error calcs
    // TODO why do I need to declare the type here?
    val modifiedWeights: List[DenseMatrix[Double]] = weights :+ DenseMatrix.eye[Double](weights.last.cols)

    val weightActivationPairs: List[(DenseMatrix[Double], DenseVector[Double])] = modifiedWeights.reverse zip activations.reverse
    println(s"weightActivationPairs: $weightActivationPairs")

    // TODO try scanRight without reversing?
    // derivative of total error with respect to pre-neuronal activation
    val nodeErrors: List[DenseVector[Double]] = weightActivationPairs.scanLeft(dcda) {
      case (upperNodeError: DenseVector[Double], (weightMatrix, activationsBelow)) => {
        {
          {
//            println(s"upperNodeError: $upperNodeError")
//            println(s"weightMatrix: $weightMatrix")
//            println(s"activationsBelow: $activationsBelow")
            val deda: DenseVector[Double] = weightMatrix * upperNodeError
            println(s"deda: $deda")
            val dadz: DenseVector[Double] = activationsBelow.map(activationFunc.derivative)
            println(s"dadz: $dadz")
            dadz :* deda
          }
        }
      }
    }.tail // drop the initial (reversed) term which only relates to the cost function level

    println(s"nodeErrors: $nodeErrors")

    val dedw: List[DenseVector[Double]] = nodeErrors.init zip activations.init.reverse map { case (errors, acts) =>
      errors :* acts
    }
    println(s"dedw: $dedw")

    val learningRate: Double = 0.5
    println(s"weights.reverse: ${weights.reverse}")


    val newWeights = weights.reverse zip dedw map { case (ws, ds) =>
      ws(*,::) - (ds * learningRate)
    }
    println(s"newWeights: $newWeights")

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
  val DefaultActivationFunc: ActivationFunc = Sigmoid
  val DefaultCostFunc: CostFunc = ResidualSumOfSquares
}
