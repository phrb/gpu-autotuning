dirpath <- "~/code/gpu-autotuning/experiments/"

#General Configurations of the graphics
cexTam=1.25
PaletteColor <- c("red", "blue", "darkgray", "orange","black","lightblue", "lightblue","violet")

#TODO
#Graficos de lineas dos LogBest Para ver cuantas vezes de 3600 são necesarias para encontrar um minimo máximo

graphics <- function(){
    setEPS()
    postscript(paste("../../../images/", app[j], "-", gpu[i],
                     "-Box.eps",sep=""),
               height = 10, width = 11)
    par(mar=c(4, 9, 1, 1) + 0.1, mgp=c(7, 1.5, 0), las=1)
    boxplot(opt0[(opt0 < 9999)], opt1[(opt1 < 9999)], opt2[(opt2 < 9999)],
            opt3[(opt3 < 9999)], benchmark[(benchmark < 9999)],
            names = c("-O0", "-O1","-O2","-O3", "Tuned"),
            ylab="Execution Time (seconds)",
            cex.axis = 2.3,
            cex.lab = 2.3
    )
    dev.off()

    postscript(paste("../../../images/", app[j], "-", gpu[i],
                     "-Best.eps",sep=""),
               height = 11, width = 11)
    par(mar=c(9, 9, 1, 1) + 0.1, mgp=c(7, 1.5, 0), las=1)
    plot(logBest[["V2"]]~logBest[["V1"]],
         type="l", col="black",
         xlab="Tunning Time (seconds)",
         ylab="Execution Time (seconds)",
         cex.main=1.43,
         cex.axis = 2.3,
         cex.lab = 2.3
         )
    points(logBest[["V2"]]~logBest[["V1"]],
           cex=3)
    dev.off()
}

setwd(paste(dirpath, sep=""))

#gpu <- c("GTX-680", "Tesla-K20", "Tesla-K40")
gpu <- c("GTX-680" ,  "Tesla-K20", "Tesla-K40")
for(i in 1:length(gpu)){
  setwd(paste(dirpath,gpu[i], sep=""))

  app <- c("MatMulGPU", "MatMulShared", "MatMulSharedUn", "MatMulUn", "SubSeqMax")

  if (gpu[i]== "Tesla-K20"){
      app <- c("MatMulGPU", "MatMulShared",  "MatMulUn", "SubSeqMax")
  }
  for(j in 1:length(app)){
    setwd(paste("./", app[j], sep=""))
    if (app[j] == "MatMulGPU" | app[j] == "MatMulShared" | app[j] == "MatMulUn"){
        opt0 <- scan(paste("./size_1024_baseline/opt_0.txt",sep=""))
        opt1 <- scan(paste("./size_1024_baseline/opt_1.txt",sep=""))
        opt2 <- scan(paste("./size_1024_baseline/opt_2.txt",sep=""))
        opt3 <- scan(paste("./size_1024_baseline/opt_3.txt",sep=""))
        benchmark <- scan(paste("./size_1024_time_3600/run_0/benchmark.txt",sep=""))
        logAll <- read.table(paste("./size_1024_time_3600/run_0/logall.txt",sep=""))
        logBest <- read.table(paste("./size_1024_time_3600/run_0/logbest.txt",sep=""))
        graphics()
    }
    if (app[j] == "MatMulSharedUn"){
        opt0 <- scan(paste("./size_256_baseline/opt_0.txt",sep=""))
        opt1 <- scan(paste("./size_256_baseline/opt_1.txt",sep=""))
        opt2 <- scan(paste("./size_256_baseline/opt_2.txt",sep=""))
        opt3 <- scan(paste("./size_256_baseline/opt_3.txt",sep=""))
        benchmark <- scan(paste("./size_256_time_3600/run_0/benchmark.txt",sep=""))
        logAll <- read.table(paste("./size_256_time_3600/run_0/logall.txt",sep=""))
        logBest <- read.table(paste("./size_256_time_3600/run_0/logbest.txt",sep=""))
        graphics()
    }
    if (app[j] == "SubSeqMax" & gpu[i] == "Tesla-K40"){
        opt0 <- scan(paste("./size_134217728_baseline/opt_0.txt",sep=""))
        opt1 <- scan(paste("./size_134217728_baseline/opt_1.txt",sep=""))
        opt2 <- scan(paste("./size_134217728_baseline/opt_2.txt",sep=""))
        opt3 <- scan(paste("./size_134217728_baseline/opt_3.txt",sep=""))
        benchmark <- scan(paste("./size_134217728_time_3600/run_0/benchmark.txt",sep=""))
        logAll <- read.table(paste("./size_134217728_time_3600/run_0/logall.txt",sep=""))
        logBest <- read.table(paste("./size_134217728_time_3600/run_0/logbest.txt",sep=""))
        graphics()
    }
    if (app[j] == "SubSeqMax" & gpu[i] == "Tesla-K20"){
        opt0 <- scan(paste("./size_1073741824_baseline/opt_0.txt",sep=""))
        opt1 <- scan(paste("./size_1073741824_baseline/opt_1.txt",sep=""))
        opt2 <- scan(paste("./size_1073741824_baseline/opt_2.txt",sep=""))
        opt3 <- scan(paste("./size_1073741824_baseline/opt_3.txt",sep=""))
        benchmark <- scan(paste("./size_1073741824_time_3600/run_0/benchmark.txt",sep=""))
        logAll <- read.table(paste("./size_1073741824_time_3600/run_0/logall.txt",sep=""))
        logBest <- read.table(paste("./size_1073741824_time_3600/run_0/logbest.txt",sep=""))
        graphics()
    }
    setwd("../")
  }
}
