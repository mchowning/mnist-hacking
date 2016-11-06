package com.mattchowning.neuralnet

import org.scalatest.{FreeSpec, Matchers}

class NeuralNetSpec extends FreeSpec with Matchers with StackBehaviors {

  "An initial neural net" - {
    behave like wellFunctioningNeuralNet(InitialNeuralNet)
  }

  "A mutable neural net" - {
    behave like wellFunctioningNeuralNet(MutableNeuralNet)
  }
}
