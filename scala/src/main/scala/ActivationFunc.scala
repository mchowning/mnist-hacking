
trait ActivationFunc {
  def apply(x: Double): Double
  def derivative(x: Double): Double
}

object Sigmoid extends ActivationFunc {
  override def apply(x: Double): Double = 1d/(1d+scala.math.exp(-x))
  override def derivative(x: Double): Double = x * (1 - x)
}
