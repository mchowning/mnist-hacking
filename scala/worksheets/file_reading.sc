import java.io.{DataInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import breeze.linalg.DenseVector

val testImagesFilename = "t10k-images-idx3-ubyte.gz"
val testLabelsFilename = "t10k-labels-idx1-ubyte.gz"
val trainImagesFilename = "train-images-idx3-ubyte.gz"
val trainLabelsFilename = "train-labels-idx1-ubyte.gz"
//val filePath = "/Users/matt/dev/machine-learning/mnist/"
val filePath = Option(System.getenv("MNIST_PATH")).getOrElse("/Users/matt/dev/machine-learning/mnist/")
val stream = new DataInputStream(new GZIPInputStream(new FileInputStream(filePath+testLabelsFilename)))

val magicNumber = stream.readInt()
val numberOfRows = stream.readInt()

val labelsAsInts = readLabels(0)

// convert labelsAsInts into a sparse binary matrix
val labelsAsVectors = labelsAsInts.map { label =>
  DenseVector.tabulate[Double](10) { i => if (i == label) 1.0 else 0.0 }
}

def readLabels(ind: Int): Stream[Int] =
  if (ind >= numberOfRows) {
    Stream.empty
  } else {
    Stream.cons(stream.readByte(), readLabels(ind+1))
  }


