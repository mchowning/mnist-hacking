import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister

class NeuralNet(layerConfig: List[Int],
                userWeights: List[DenseMatrix[Double]] = Nil,
               // TODO extract out ActivationFunc
                activationF: Double => Double = { x => 1d/(1d+scala.math.exp(-x)) }, // sigmoid
                dActivationF: Double => Double = { x =>  x * (1 - x)},
                // TODO extract out CostFunc
                costF: (DenseVector[Double], DenseVector[Double]) => Double = {
                  // TODO double check that I have expected and actual in the right order for RSS
                  (expected, actual) => sum((expected - actual) :^ 2d) / 2 // RSS
                },
                dCostF: (Double, Double) => Double = {
                  (expected, actual) =>  actual - expected
                }) {


  val weights = if (userWeights != Nil) userWeights
                else randomWeights

  def randomWeights = {
    println("Initializing with random weights...")
    // List of tuples where each tuple represents (# input nodes, # output nodes) from one layer to the next
    val transitions = layerConfig zip layerConfig.tail
    // TODO use 3d matrix?
    transitions.map { case (i, j) =>
      // adding one node to the i layer to account for the bias node
      // (not adding to the j layer because no weights go into the bias)
      DenseMatrix.rand(i, j, new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0))).uniform)
      DenseMatrix.rand(i+1, j, new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0))).uniform)
      //    DenseMatrix.rand(i+1, j)
      //    DenseMatrix.rand(i, j)
    }
  }

  // TODO use Streams

  def totalError(inputs: List[Double], expected: DenseVector[Double]): Double = {
    val output = forwardPropagate(inputs)
    costF(expected, output)
  }

  def train(inputs: List[Double], expected: DenseVector[Double]) = {
    val output: DenseVector[Double] = forwardPropagate(inputs)
    val cost = costF(expected, output)

    // TODO backprop


  }

  def forwardPropagate(inputs: List[Double]): DenseVector[Double] = {
    checkInputs(inputs)

    val inputVals = DenseVector(inputs.toArray)
    weights.foldLeft(inputVals) { (iVals, wVals) =>

      // add in bias units
      // TODO find better way to add to a vector
      val layerNodes = DenseVector(iVals.toArray :+ 1d)

//      println(s"layerNodes: $layerNodes")
//      println(s"wVals: $wVals")

      (layerNodes.t * wVals).t  // multiply weights and inputs
        .map(activationF)   // run results through activation function
//        .map { x =>
//          println(s"z-value: $x")
//          val a = activationF(x)
//          println(s"a-value: $a")
//          a
//        }
    }
  }

  private def checkInputs(inputs: List[Double]): Unit = {
    // subtract 1 to account for the bias weight
    val numInputs = weights.head.rows - 1
    if (numInputs != inputs.size) {
      throw new Exception("InvalidNetworkInputs: This neural network requires " + numInputs + " inputs")
      // throw new Exception(String.format("Weights of size %s and inputs of size %s do not match",
      //                                   Integer.toString(weights.size),
      //                                   Integer.toString(inputs.size)))
    }
  }
}
