package com.mattchowning

import breeze.linalg.DenseVector

object Main extends App {
  println("Hello, world")
  val x = DenseVector(0.0, 0.0)
  println(x)
  println(x)
  println(x)
  println(x)

  def bar = 1.0
  def l = List(1.0, 2.0)
  println(bar)

  def myMethod() = {
    println("os1")
    println("os2")
    println("osm")
  }
}

