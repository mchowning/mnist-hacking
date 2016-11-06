package com.mattchowning.neuralnet

import breeze.linalg.{DenseMatrix, DenseVector}
import com.mattchowning.{ResidualSumOfSquares, Sigmoid}
import org.scalatest.FreeSpec

trait StackBehaviors { this: FreeSpec =>

  def wellFunctioningNeuralNet(neuralNetFactory: NeuralNetFactory) = {

    "with weights based on https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/" - {
      val nn = neuralNetFactory.withWeights(weights = List(DenseMatrix((0.15, 0.25), (0.20, 0.30), (0.35, 0.35)),
                                                            DenseMatrix((0.40, 0.50), (0.45, 0.55), (0.6, 0.6))),
                                             activationFunc = Sigmoid,
                                             costFunc = ResidualSumOfSquares,
                                             learningRate = 0.5)
      val inputs: List[Double] = List(0.05, 0.10)
      val expectedOutput: DenseVector[Double] = DenseVector(0.01, 0.99)

      "should forward propagate" in {
        val outputs = nn.forwardPropagate(inputs)
        assert(outputs == DenseVector(0.7513650695523157, 0.7729284653214625))
      }

      "should calculate total error" in {
        val error = nn.totalError(inputs, expectedOutput)
        assert(error == 0.2983711087600027)
      }

      "should backpropagate the weights" in {
        val newWeights = nn.train(inputs, expectedOutput)
        assert(newWeights == List(DenseMatrix((0.1497807161327628, 0.24950228726473914),
                                              (0.19978071613276283, 0.29950228726473915),
                                              (0.3497807161327628, 0.34950228726473914)),
                                  DenseMatrix((0.35891647971788465, 0.5113701211079891),
                                               (0.40891647971788464, 0.5613701211079891),
                                               (0.5589164797178846, 0.611370121107989))))
      }
    }

    "with valid weights should not throw an exception" in {
      neuralNetFactory.withWeights(List(DenseMatrix.ones(1, 2),
                                        DenseMatrix.ones(3, 3),
                                        DenseMatrix.ones(4, 1)))
    }

    "with invalid weights should throw an exception" in {
      assertThrows[Exception] {
        neuralNetFactory.withWeights(List(DenseMatrix.ones(1, 2),
                                          DenseMatrix.ones(2, 1)))
      }
    }
  }
}
