package com.mattchowning.neuralnet

import breeze.linalg.{DenseMatrix, DenseVector}

trait NeuralNet {
  def totalError(inputs: List[Double], expected: DenseVector[Double]): Double
  def train(inputs: List[Double], expected: DenseVector[Double]): List[DenseMatrix[Double]]
  def forwardPropagate(inputs: List[Double]): DenseVector[Double]
}
