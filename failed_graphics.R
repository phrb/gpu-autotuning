dirpath <- "~/code/gpu-autotuning/experiments/"

#General Configurations of the graphics
cexTam=1.25
PaletteColor <- c("red", "blue", "darkgray", "orange","black","lightblue", "lightblue","violet")

results_summary <- function(){
    setEPS()
    postscript(paste("../FailedSummary.eps",sep=""),
               height = 10, width = 11)
    par(mar=c(4, 9, 1, 1) + 0.1, mgp=c(7, 1.5, 0), las=1)
    # MatMulGPU
    gtx   <- read.table(paste("./GTX-680/MatMulGPU/size_1024_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
    k20   <- read.table(paste("./Tesla-K20/MatMulGPU/size_1024_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
    k40   <- read.table(paste("./Tesla-K40/MatMulGPU/size_1024_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]

    matmulgpu <- c(gtx , k20, k40)

    # MatMulUn
    gtx   <- read.table(paste("./GTX-680/MatMulUn/size_1024_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
    k20   <- read.table(paste("./Tesla-K20/MatMulUn/size_1024_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
    k40   <- read.table(paste("./Tesla-K40/MatMulUn/size_1024_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
    
    matmulun <- c(gtx, k20, k40)

    # MatMulShared
    gtx   <- read.table(paste("./GTX-680/MatMulShared/size_1024_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
    k20   <- read.table(paste("./Tesla-K20/MatMulShared/size_1024_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
    k40   <- read.table(paste("./Tesla-K40/MatMulShared/size_1024_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
     
    matmulshared <- c(gtx, k20, k40)

    # MatMulSharedUn
    gtx   <- read.table(paste("./GTX-680/MatMulSharedUn/size_256_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
#   k20   <- read.table(paste("./Tesla-K20/MatMulSharedUn/size_1024_time_3600/run_0/failed_stats.txt",sep=""))
    k40   <- read.table(paste("./Tesla-K40/MatMulSharedUn/size_256_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
    
    matmulsharedun <- c(gtx, 0, k40)

    # SubSeqMax
    gtx   <- read.table(paste("./GTX-680/SubSeqMax/size_134217728_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
    k20   <- read.table(paste("./Tesla-K20/SubSeqMax/size_134217728_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
    k40   <- read.table(paste("./Tesla-K40/SubSeqMax/size_134217728_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
     
    subseqmax <- c(gtx, k20, k40)

    print(matmulun)
    print(matmulgpu)
    print(matmulshared)
    print(matmulsharedun)
    print(subseqmax)

    final <- data.frame(MMU=100*matmulun, MMG=100*matmulgpu, MMSU=100*matmulsharedun, MMS=100*matmulshared, SSM=100*subseqmax)

    barplot(as.matrix(final),
            ylab="Percentage of Failed Configurations",
            beside=T,
            ylim=c(0, 12),
            xpd=F,
            col=gray.colors(3, start=0, end=1),
            cex.names = 2.3,
            space=c(0,0.3),
            names=c("#1", "#2", "#3", "#4", "Sub-Array"),
            cex.axis = 2.3,
            cex.lab = 2.3
    )
    legend(7, 12, c("GTX-680", "Tesla-K20", "Tesla-K40"), fill=gray.colors(3, start=0, end=1), cex=2)
    dev.off()
}

setwd(paste(dirpath, sep=""))
results_summary()
