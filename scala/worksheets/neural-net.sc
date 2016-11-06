import breeze.linalg.{DenseMatrix, DenseVector}
import com.mattchowning.neuralnet.InitialNeuralNet

val nn = InitialNeuralNet.withWeights(weights = List(DenseMatrix((0.15, 0.25), (0.20, 0.30), (0.35, 0.35)),
                                              DenseMatrix((0.40, 0.50), (0.45, 0.55), (0.6, 0.6))))

//var nn = tes.neuralnet.NeuralNet.withLayers(List(2,2,2))

val inputs: List[Double] = List(0.05, 0.10)
val expected: DenseVector[Double] = DenseVector(0.01, 0.99)

nn.train(inputs, expected)

//
//var oldError = 999999d
//val nTimes = 10
//1 to nTimes foreach { n =>
//  nn.train(inputs, expected)
//  println(s"error: ${nn.totalError(inputs, expected)}")
//  if (oldError < error) {
//    println(s"oldError: $oldError")
//    println(s"error: $error")
//  }
//  val newWeights = nn.train(inputs, expected)
//  nn = tes.neuralnet.NeuralNet.withWeights(newWeights)
//  if (n == nTimes) println(s"last error:$n $error")
//}
//println("done")

List(10,100,1000).scanRight(0) {case (a,b) => println(s"$a, $b"); a+b}