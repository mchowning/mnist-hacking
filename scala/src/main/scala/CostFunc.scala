import breeze.linalg.{DenseVector, sum}

trait CostFunc {
  def apply(expected: DenseVector[Double], actual: DenseVector[Double]): Double
  // TODO should this be using vectors?
  def derivative(expected: Double, actual: Double): Double
}

object ResidualSumOfSquares extends CostFunc {
  override def apply(expected: DenseVector[Double], actual: DenseVector[Double]): Double =
    sum((expected - actual) :^ 2d) / 2

  override def derivative(expected: Double, actual: Double): Double =
    actual - expected
}
