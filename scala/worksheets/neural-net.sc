import breeze.linalg.DenseMatrix
import com.mattchowning.neuralnet.InitialNeuralNet$

val nn = InitialNeuralNet.withWeights(List(DenseMatrix.ones(1,1),
                                    DenseMatrix.ones(2,1)))

//val nn = tes.neuralnet.NeuralNet.withWeights(weights = List(DenseMatrix((0.15, 0.25), (0.20, 0.30), (0.35, 0.35)),
//                                              DenseMatrix((0.40, 0.50), (0.45, 0.55), (0.6, 0.6))))

//var nn = tes.neuralnet.NeuralNet.withLayers(List(2,2,2))

//val inputs: List[Double] = List(0.05, 0.10)
//val expected: DenseVector[Double] = DenseVector(0.01, 0.99)
//
//var oldError = 999999d
//val nTimes = 10000
//1 to nTimes foreach { n =>
//  val error: Double = nn.totalError(inputs, expected)
//  if (oldError < error) {
//    println(s"oldError: $oldError")
//    println(s"error: $error")
//  }
//  val newWeights = nn.train(inputs, expected)
//  nn = tes.neuralnet.NeuralNet.withWeights(newWeights)
//  if (n == nTimes) println(s"last error:$n $error")
//}
//println("done")

