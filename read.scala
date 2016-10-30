import java.io.{ File, FileInputStream, FileOutputStream, DataInputStream }
import java.util.zip.GZIPInputStream

val stream = new DataInputStream(new GZIPInputStream(new FileInputStream("t10k-images-idx3-ubyte.gz")))
for (a <- 1 to 10) {
  System.out.println(a)
  System.out.println(stream.readInt())
}
