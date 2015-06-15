dirpath <- "~/code/gpu-autotuning/experiments/"

#General Configurations of the graphics
cexTam=1.25
PaletteColor <- c("red", "blue", "darkgray", "orange","black","lightblue", "lightblue","violet")

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

calc_speedup <- function(opt, values){
    return ((1 - (median(values[values < 9999]) / median(opt[opt < 9999])) ) * 100)
}

results_summary <- function(){
    setEPS()
    postscript(paste("../images/Summary.eps",sep=""),
               height = 10, width = 11)
    par(mar=c(4, 9, 1, 1) + 0.1, mgp=c(7, 1.5, 0), las=1)
    # MatMulGPU
    gtx   <- scan(paste("./GTX-680/MatMulGPU/size_1024_time_3600/run_0/benchmark.txt",sep=""))
    k20   <- scan(paste("./Tesla-K20/MatMulGPU/size_1024_time_3600/run_0/benchmark.txt",sep=""))
    k40   <- scan(paste("./Tesla-K40/MatMulGPU/size_1024_time_3600/run_0/benchmark.txt",sep=""))
    o2gtx <- scan(paste("./GTX-680/MatMulGPU/size_1024_baseline/opt_2.txt",sep=""))
    o2k20 <- scan(paste("./Tesla-K20/MatMulGPU/size_1024_baseline/opt_2.txt",sep=""))
    o2k40 <- scan(paste("./Tesla-K40/MatMulGPU/size_1024_baseline/opt_2.txt",sep=""))

    matmulgpu <- c(calc_speedup(o2gtx, gtx), calc_speedup(o2k20, k20), calc_speedup(o2k40, k40))

    # MatMulUn
    gtx   <- scan(paste("./GTX-680/MatMulUn/size_1024_time_3600/run_0/benchmark.txt",sep=""))
    k20   <- scan(paste("./Tesla-K20/MatMulUn/size_1024_time_3600/run_0/benchmark.txt",sep=""))
    k40   <- scan(paste("./Tesla-K40/MatMulUn/size_1024_time_3600/run_0/benchmark.txt",sep=""))
    o2gtx <- scan(paste("./GTX-680/MatMulUn/size_1024_baseline/opt_2.txt",sep=""))
    o2k20 <- scan(paste("./Tesla-K20/MatMulUn/size_1024_baseline/opt_2.txt",sep=""))
    o2k40 <- scan(paste("./Tesla-K40/MatMulUn/size_1024_baseline/opt_2.txt",sep=""))
    
    matmulun <- c(calc_speedup(o2gtx, gtx), calc_speedup(o2k20, k20), calc_speedup(o2k40, k40))
    print(calc_speedup(o2gtx, gtx))

    # MatMulShared
    gtx   <- scan(paste("./GTX-680/MatMulShared/size_1024_time_3600/run_0/benchmark.txt",sep=""))
    k20   <- scan(paste("./Tesla-K20/MatMulShared/size_1024_time_3600/run_0/benchmark.txt",sep=""))
    k40   <- scan(paste("./Tesla-K40/MatMulShared/size_1024_time_3600/run_0/benchmark.txt",sep=""))
    o2gtx <- scan(paste("./GTX-680/MatMulShared/size_1024_baseline/opt_2.txt",sep=""))
    o2k20 <- scan(paste("./Tesla-K20/MatMulShared/size_1024_baseline/opt_2.txt",sep=""))
    o2k40 <- scan(paste("./Tesla-K40/MatMulShared/size_1024_baseline/opt_2.txt",sep=""))
     
    matmulshared <- c(calc_speedup(o2gtx, gtx), calc_speedup(o2k20, k20), calc_speedup(o2k40, k40))

    # MatMulSharedUn
    gtx   <- scan(paste("./GTX-680/MatMulSharedUn/size_256_time_3600/run_0/benchmark.txt",sep=""))
#   k20   <- scan(paste("./Tesla-K20/MatMulSharedUn/size_1024_time_3600/run_0/benchmark.txt",sep=""))
    k40   <- scan(paste("./Tesla-K40/MatMulSharedUn/size_256_time_3600/run_0/benchmark.txt",sep=""))
    o2gtx <- scan(paste("./GTX-680/MatMulSharedUn/size_256_baseline/opt_2.txt",sep=""))
#   o2k20 <- scan(paste("./Tesla-K20/MatMulSharedUn/size_1024_baseline/opt_2.txt",sep=""))
    o2k40 <- scan(paste("./Tesla-K40/MatMulSharedUn/size_256_baseline/opt_2.txt",sep=""))
    
    matmulsharedun <- c(calc_speedup(o2gtx, gtx), 0, calc_speedup(o2k40, k40))

    # SubSeqMax
    gtx   <- scan(paste("./GTX-680/SubSeqMax/size_1073741824_time_3600/run_0/benchmark.txt",sep=""))
    k20   <- scan(paste("./Tesla-K20/SubSeqMax/size_1073741824_time_3600/run_0/benchmark.txt",sep=""))
    k40   <- scan(paste("./Tesla-K40/SubSeqMax/size_134217728_time_3600/run_0/benchmark.txt",sep=""))
    o2gtx <- scan(paste("./GTX-680/SubSeqMax/size_1073741824_baseline/opt_2.txt",sep=""))
    o2k20 <- scan(paste("./Tesla-K20/SubSeqMax/size_1073741824_baseline/opt_2.txt",sep=""))
    o2k40 <- scan(paste("./Tesla-K40/SubSeqMax/size_134217728_baseline/opt_2.txt",sep=""))
     
    subseqmax <- c(calc_speedup(o2gtx, gtx), calc_speedup(o2k20, k20), calc_speedup(o2k40, k40))

    final <- data.frame(MMU=matmulun, MMG=matmulgpu, MMSU=matmulsharedun, MMS=matmulshared, SSM=subseqmax)

    barplot(as.matrix(final),
            ylab="Percentage of Speedup",
            beside=T,
            ylim=c(-2.8, 30),
            xpd=F,
            col=gray.colors(3, start=0, end=1),
            cex.names = 2.3,
            space=c(0,0.3),
            names=c("#1", "#2", "#3", "#4", "Sub-Array"),
            cex.axis = 2.3,
            cex.lab = 2.3
    )
    legend(4, 25, c("GTX-680", "Tesla-K20", "Tesla-K40"), fill=gray.colors(3, start=0, end=1), cex=2)
    dev.off()
}

setwd(paste(dirpath, sep=""))
results_summary()

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
