import breeze.linalg.{DenseMatrix, DenseVector}
import org.scalatest._


class NeuralNetSpec extends FreeSpec with Matchers {

  "A neural network based on https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/" - {
    val nn = new NeuralNet(numNodesPerLayer = List(2,2,2),
                            userWeights = List(DenseMatrix((0.15, 0.25), (0.20, 0.30), (0.35, 0.35)),
                                                DenseMatrix((0.40, 0.50), (0.45, 0.55), (0.6, 0.6))),
                            activationFunc = Sigmoid,
                            costFunc = ResidualSumOfSquares)

    "should forward propagate" in {
      val outputs = nn.forwardPropagate(List(0.05, 0.10))
      outputs shouldEqual DenseVector(0.7513650695523157, 0.7729284653214625)
    }

    "should calculate total error" in {
      val error = nn.totalError(List(0.05, 0.10), DenseVector(0.01, 0.99))
      error shouldEqual 0.2983711087600027
    }
  }
}
