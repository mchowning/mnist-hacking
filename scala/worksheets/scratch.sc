import breeze.linalg.{DenseVector, sum}

def costF: (DenseVector[Double], DenseVector[Double]) => Double = {
  (expected, actual) => {
    val diff: DenseVector[Double] = expected - actual
//    val sqDiff: DenseVector[Double] = diff :^ 2d
//    val totalSqDiff: Double = sum(sqDiff)
//    totalSqDiff / 2
    sum(diff :^ 2d ) / 2
  } // RSS
}

costF(DenseVector(0.751, 0.772), DenseVector(0.01, 0.99))

