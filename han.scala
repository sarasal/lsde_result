
import org.apache.hadoop.io.Text
import org.apache.hadoop.io.BytesWritable
import org.apache.hadoop.io.compress.GzipCodec
import com.cotdp.hadoop.ZipFileInputFormat

import org.apache.pdfbox.pdfparser.PDFParser
import org.apache.pdfbox.text.PDFTextStripper
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.pdfbox.io.RandomAccessBuffer

import com.optimaize.langdetect.profiles.LanguageProfileReader
import com.optimaize.langdetect.LanguageDetectorBuilder
import com.optimaize.langdetect.ngram.NgramExtractors
import com.optimaize.langdetect.text.CommonTextObjectFactories

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object pdf {
def extract_text(bytes: BytesWritable): String = {
    try {
	val pdfStripper = new PDFTextStripper;
        val parser = new PDFParser(new RandomAccessBuffer(bytes.getBytes))
        parser.parse()
        return pdfStripper.getText(new PDDocument(parser.getDocument));
    } catch {
         case e: Exception =>
    }
    return ""
}

def main(args:Array[String]){
//conf = SparkConf().setAppName("pdf Application")
val sc: SparkContext  = new SparkContext(new SparkConf())
val spark = SparkSession.builder.appName("pdf Application").getOrCreate()

val zipcontent = sc.newAPIHadoopFile(
        "/user/hannesm/lsde/scihub/91*.zip",
        classOf[ZipFileInputFormat],
        classOf[Text],
        classOf[BytesWritable])

//def extract_text(bytes: BytesWritable): String = {
//    try { 
//        val pdfStripper = new PDFTextStripper;
//        val parser = new PDFParser(new RandomAccessBuffer(bytes.getBytes))
//        parser.parse()
//        return pdfStripper.getText(new PDDocument(parser.getDocument));
//    } catch {
//         case e: Exception => 
//    }
//    return ""
//}

val documents = zipcontent.map(tuple => (tuple._1.toString(), extract_text(tuple._2)))

val doi_pattern = "^\\d+/(.+)\\.pdf$".r
val clean = documents.map(t => (t._1 match {case doi_pattern(el) => el case _ => ""}, 
   t._2.filter(_ >= ' ').map(c => if(c == '\t') ' ' else c))).filter(t => t._2.length() > 100)


val langs = clean.mapPartitions((partition: Iterator[Tuple2[String, String]]) => {
  val ld = LanguageDetectorBuilder.create(NgramExtractors.standard).withProfiles(new LanguageProfileReader().readAllBuiltIn).build
  val tof = CommonTextObjectFactories.forDetectingOnLargeText

  partition.flatMap((text: Tuple2[String, String]) => {
    val lang = ld.detect(tof.forText(text._2))
    var res = "NA"
    if (lang.isPresent)
       res = lang.get.getLanguage
    Some((res, text._1, text._2))
   })
})

val enonly = langs.filter(t => t._1 == "en").map(t => (t._2,t._3))
val df= spark.createDataFrame(enonly).toDF("doi","text")
df.write.parquet("/user/lsde22/zipfiles0_out")

//println("documents")
//documents.take(1).foreach(println)

//println("clean")
//clean.take(1).foreach(println)

//println("enonly")
//enonly.take(1).foreach(println)

//println("df")
//df.show()
spark.stop()
}
}
