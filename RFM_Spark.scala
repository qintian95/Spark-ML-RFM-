package com.tq.KMeans

import java.sql.DriverManager
import java.util.{Date, Properties}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * @author tq
  *         2019/03/26
  */
object RFM_Spark {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val spark = SparkSession.builder().appName("test").master("local").config("spark.debug.maxToStringFields", "100").config("spark.sql.crossJoin.enabled", "true").getOrCreate()

    val date1 = System.currentTimeMillis()

    spark.read.option("header", "true").csv("data/customer.csv").select("customer_id", "total_cost_fee", "total_repair_times", "after_days").createTempView("v1")


    spark.sql("select customer_id,sum(total_cost_fee) total_cost_fee,sum(total_repair_times) total_repair_times,min(cast(after_days as int)) after_days from v1 where total_cost_fee>0  group by customer_id").createTempView("t1")

    //spark.sql("select * from t1 where customer_id='1001mm_1195'").show()


    //    spark.sql("""select max(total_cost_fee) max_total_cost_fee,min(total_cost_fee) min_total_cost_fee,
    //                                  max(total_repair_times) max_total_repair_times, min(total_repair_times) min_total_repair_times,
    //                                  3502 as max_after_days, 7 as min_after_days from t1""").createTempView("t2")

    spark.sql(
      """select max(total_cost_fee) max_total_cost_fee,min(total_cost_fee) min_total_cost_fee,
                                  max(total_repair_times) max_total_repair_times, min(total_repair_times) min_total_repair_times,
                                   max(after_days) as max_after_days, min(after_days) as min_after_days from t1""").createTempView("t2")

    val max_total_cost_fee = spark.sql("select max_total_cost_fee from t2").collect().head.toString().split("\\[")(1).split("\\]")(0).toDouble.toInt
    val min_total_cost_fee = spark.sql("select min_total_cost_fee from t2").collect().head.toString().split("\\[")(1).split("\\]")(0).toDouble.toInt
    val max_total_repair_times = spark.sql("select max_total_repair_times from t2").collect().head.toString().split("\\[")(1).split("\\]")(0).toDouble.toInt
    val min_total_repair_times = spark.sql("select min_total_repair_times from t2").collect().head.toString().split("\\[")(1).split("\\]")(0).toDouble.toInt
    val max_after_days = spark.sql("select max_after_days from t2").collect().head.toString().split("\\[")(1).split("\\]")(0).toDouble.toInt
    val min_after_days = spark.sql("select min_after_days from t2").collect().head.toString().split("\\[")(1).split("\\]")(0).toDouble.toInt

    val driver = "com.mysql.jdbc.Driver"
    val url = "jdbc:mysql://aa.murfon.com:3306/automotiveaftermarket?useSSL=false"
    val user = "root"
    val password = "~!@Murfon@2019"
    Class.forName(driver)
    val con = DriverManager.getConnection(url, user, password)
    val statement = con.createStatement()


    // spark.sql("select * from t2 ").show()
    val data = spark.sql(
      """select t1.customer_id,
        |(t1.total_cost_fee - t2.min_total_cost_fee)/(t2.max_total_cost_fee - t2.min_total_cost_fee) as total_cost_fee,
        |(t1.total_repair_times - t2.min_total_repair_times)/(t2.max_total_repair_times - t2.min_total_repair_times) as total_repair_times,
        |(t1.after_days - t2.min_after_days)/(t2.max_after_days - t2.min_after_days) as after_days
        |from
        |  t1 ,
        |  t2
      """.stripMargin)


    val schema = data.schema.map(a => s"${a.name}").drop(1)
    val assembler = new VectorAssembler().setInputCols(schema.toArray).setOutputCol("features")

    //  val data2 = data.na.fill(-1e9)
    val userProfile = assembler.transform(data).select("customer_id", "features")




//
    //    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)
    //    val scalerModel = scaler.fit(userProfile)
    //    scalerModel.transform(userProfile)

    //    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures")
    //    val minMaxScalerModel = scaler.fit(userProfile)
    //    val userProfileModel= minMaxScalerModel.transform(userProfile)

    val Array(training, test) = userProfile.randomSplit(Array(0.8, 0.2))
    training.createTempView("v2")
    test.createTempView("v3")

    val trainingModel = spark.sql("select customer_id, features from v2")
    val testModel = spark.sql("select customer_id, features from v3")

    //    val trainingModel = spark.sql("select customer_id,scaledFeatures features from v2")
    //    val testModel = spark.sql("select customer_id,scaledFeatures features from v3")
    //trainingModel.show()

    /*val arr: Array[Double] = Array( 7,9,10,12,15,20)
    for (i <- 0 until arr.length - 1) {

      val kmeans = new KMeans().setK(8).setMaxIter(30).setSeed(arr(i).toInt).setFeaturesCol("features").setTol(1e-6)
      val modelUsers = kmeans.fit(userProfile)
      val evaluator = new ClusteringEvaluator()
      println(s"第$i 次WSSSE=${modelUsers.computeCost(userProfile)} ")
      modelUsers.clusterCenters.foreach(a => {
        var id = a.hashCode()
        println(s"'$id',${(a(0) * (max_total_cost_fee - min_total_cost_fee) + min_total_cost_fee).toInt},${(a(1) * (max_total_repair_times - min_total_repair_times) + min_total_repair_times).toInt},${(a(2) * (max_after_days - min_after_days) + min_after_days).toInt}")

      }

      )


      val dataModel: DataFrame = modelUsers.transform(userProfile)
      val silhouette: Double = evaluator.evaluate(dataModel)
      println(s"silhouette=$silhouette")
      println("==================================================")
    }*/

    val kmeans = new KMeans().setK(8).setMaxIter(20).setFeaturesCol("features").setTol(1e-8).setSeed(7)
    val modelUsers = kmeans.fit(userProfile)
    val evaluator = new ClusteringEvaluator()


    println(s"WSSSE=${modelUsers.computeCost(userProfile)} ")
    modelUsers.clusterCenters.foreach(a => {
      var id = a.hashCode()
      println(s"'$id',${(a(0) * (max_total_cost_fee - min_total_cost_fee) + min_total_cost_fee).toInt},${(a(1) * (max_total_repair_times - min_total_repair_times) + min_total_repair_times).toInt},${(a(2) * (max_after_days - min_after_days) + min_after_days).toInt}")

    }
    )


    val dataModel: DataFrame = modelUsers.transform(userProfile)
    val silhouette: Double = evaluator.evaluate(dataModel)
    println(s"silhouette=$silhouette")
    //  dataModel.orderBy("customer_id").show()


    dataModel.createTempView("t4")

    val prop = new Properties()
    prop.setProperty("user", user)
    prop.setProperty("password", password)

    spark.sql("select t4.customer_id,t1.total_cost_fee as monetary,t1.total_repair_times as frequency,t1.after_days as recency,t4.prediction as cluster from t4 join  t1 on t4.customer_id=t1.customer_id")
      .write.option("driver", driver).mode("overwrite")
      .jdbc(url, "rfm_persona", prop) //.show()

    modelUsers.clusterCenters.foreach(a => {

      var i = a.hashCode()
      var x = (a(0) * (max_total_cost_fee - min_total_cost_fee) + min_total_cost_fee).toInt
      var y = (a(1) * (max_total_repair_times - min_total_repair_times) + min_total_repair_times).toInt
      var z = (a(2) * (max_after_days - min_after_days) + min_after_days).toInt
      statement.execute(s"insert into rfm_persona values('$i',$x,$y,$z,8)")

    })
    con.close()
    println(System.currentTimeMillis() - date1)
    spark.close()
  }
}
