import breeze.linalg.{DenseMatrix, DenseVector}

val nn = new NeuralNet(numNodesPerLayer = List(2,2,2),
                       userWeights = List(DenseMatrix((0.15, 0.25), (0.20, 0.30), (0.35, 0.35)),
                         DenseMatrix((0.40, 0.50), (0.45, 0.55), (0.6, 0.6))))

val inputs: List[Double] = List(0.05, 0.10)
val expected: DenseVector[Double] = DenseVector(0.01, 0.99)
val nnError = nn.totalError(inputs, expected) // 0.298371109
//val nnResult = nn.forwardPropagate(List(0.05, 0.10))

val matrix = DenseMatrix((1,2,3), (10, 20, 30), (100, 200, 300))

val as: DenseVector[Double] = DenseVector(0.75136507, 0.772928465)
val ys: DenseVector[Double] = DenseVector(0.01, 0.99)

nn.train(inputs, expected)

