import breeze.linalg.{DenseVector, sum}

trait CostFunc {
  def apply(expected: DenseVector[Double], actual: DenseVector[Double]): Double
  def derivative(expected: DenseVector[Double], actual: DenseVector[Double]): DenseVector[Double]
}

object ResidualSumOfSquares extends CostFunc {
  override def apply(expected: DenseVector[Double], actual: DenseVector[Double]): Double =
    sum((expected - actual) :^ 2d) / 2

  override def derivative(expected: DenseVector[Double], actual: DenseVector[Double]): DenseVector[Double] =
    actual - expected
}
