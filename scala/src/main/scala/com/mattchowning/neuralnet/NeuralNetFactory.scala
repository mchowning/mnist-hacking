package com.mattchowning.neuralnet

import breeze.linalg.DenseMatrix
import com.mattchowning.{ActivationFunc, CostFunc, ResidualSumOfSquares, Sigmoid}

trait NeuralNetFactory {

  protected val DefaultActivationFunc = Sigmoid
  protected val DefaultCostFunc = ResidualSumOfSquares
  protected val DefaultLearningRate: Double = 0.5

  def withLayers(numNodesPerLayer:  List[Int],
                 activationFunc:    ActivationFunc  = DefaultActivationFunc,
                 costFunc:          CostFunc        = DefaultCostFunc,
                 learningRate:      Double          = DefaultLearningRate) = {

    println("Initializing with random weights...")
    // Adding 1 to each 'input' layer to get weights for bias node (not adding to 'output' layer because
    // bias nodes take no input)
    val randomWeights = (numNodesPerLayer zip numNodesPerLayer.tail) map {
      t => DenseMatrix.rand[Double](t._1 + 1, t._2)
      // t => DenseMatrix.rand[Double](t._1, t._2, new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0))).uniform)
    }

    withWeights(randomWeights, activationFunc, costFunc, learningRate)
  }

  def withWeights(weights:        List[DenseMatrix[Double]],
                  activationFunc: ActivationFunc  = DefaultActivationFunc,
                  costFunc:       CostFunc        = DefaultCostFunc,
                  learningRate:   Double          = DefaultLearningRate): NeuralNet
}
