name := "mnist-scala"

version := "1.0"

scalaVersion in ThisBuild := "2.11.8"

// breeze
libraryDependencies  ++= Seq(
  // other dependencies here
  "org.scalanlp" %% "breeze" % "0.12",
  // native libraries are not included by default. add this if you want them (as of 0.7)
  // native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.12",
  // the visualization library is distributed separately as well.
  // It depends on LGPL code.
  "org.scalanlp" %% "breeze-viz" % "0.12"
)

// deeplearning4j
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.6.0"
//nd4j
classpathTypes += "maven-plugin"
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.4.0"
libraryDependencies += "org.slf4j" % "slf4j-jdk14" % "1.7.21"

// ScalaTest
libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.0"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % "test"

resolvers ++= Seq(
  // if you want to use snapshot builds (currently 0.12-SNAPSHOT), use this.
  // "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
