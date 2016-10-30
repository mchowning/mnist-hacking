import breeze.linalg.{DenseMatrix, DenseVector, sum}

val nn = new NeuralNet(numNodesPerLayer = List(2,2,2),
//                       userWeights = List(DenseMatrix((0.15, 0.25, 0.35), (0.20, 0.30, 0.35)),
//                                      DenseMatrix((0.40, 0.50, 0.6), (0.45, 0.55, 0.6))))
                       userWeights = List(DenseMatrix((0.15, 0.25), (0.20, 0.30), (0.35, 0.35)),
                         DenseMatrix((0.40, 0.50), (0.45, 0.55), (0.6, 0.6))))

val nnError = nn.totalError(List(0.05, 0.10), DenseVector(0.01, 0.99)) // 0.298371109
val nnResult = nn.forwardPropagate(List(0.05, 0.10))

val matrix = DenseMatrix((1,2,3), (10, 20, 30), (100, 200, 300))

val as: DenseVector[Double] = DenseVector(0.75136507, 0.772928465)
val ys: DenseVector[Double] = DenseVector(0.01, 0.99)

def rss(a: DenseVector[Double], y: DenseVector[Double]): Double = {
  val diff: DenseVector[Double] = y - a
  0.5 * sum(diff :^ 2.0)
}
rss(as, ys)

def drss(a: DenseVector[Double], y: DenseVector[Double]): DenseVector[Double] = {
    a - y
}
val dCost = drss(as, ys)

def dacc(a:DenseVector[Double]): DenseVector[Double] = {
//  a * (1.0 - a)
  a :* (1.0 - a)
}
val dAcc = dacc(as)

val weights: DenseMatrix[Double] = DenseMatrix((0.4, 0.5), (0.45, 0.55))
val previousLayerOutput: DenseVector[Double] = DenseVector(0.5932, 0.596)
//val totalOutputChangeWithRespectToWeights =


val learningRate: Double = 0.5
//                                          dCost           dAcc          dWeight
val newW5 = weights(0,0) - learningRate * dCost(0) * dAcc(0) * 0.59326992 // .3589
val newW6 = weights(1,0) - learningRate * dCost(0) * dAcc(0) * 0.59326992 // .4086
val newW7 = weights(0,1) - learningRate * dCost(1) * dAcc(1) * 0.596884378 // .5113
val newW8 = weights(1,1) - learningRate * dCost(1) * dAcc(1) * 0.596884378 // .5613


//(-0.21707 * 0.1755100 * 0.45)
