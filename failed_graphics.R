dirpath <- "./experiments/"


#General Configurations of the graphics
cexTam=1.25
PaletteColor <- c("red", "blue", "darkgray", "orange","black","lightblue", "lightblue","violet")

results_summary <- function(){
    
    app <- c("MatMulGPU", "MatMulShared", "MatMulSharedUn", "MatMulUn", "SubSeqMax", "Bitonic", "Quicksort", "VecAdd")
    for(j in 1:length(app)){
        
        if (app[j] == "MatMulGPU" | app[j] == "MatMulShared" | app[j] == "MatMulUn" | app[j] == "MatMulSharedUn"){
            size <- 4096
        }        
        if (app[j] == "SubSeqMax"){
            size <- 536870912
        }        
        if (app[j] == "Bitonic"){
            size <- 524288
        }
        if (app[j] == "Quicksort"){
            size <- 32768
        }
        if (app[j] == "VecAdd"){
            size <- 131072
        }
        gtx   <- read.table(paste("./GTX-980/", app[j], "/size_", size, "_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
        k20   <- read.table(paste("./GTX-750/", app[j], "/size_", size, "_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
        k40   <- read.table(paste("./Tesla-K40/", app[j], "/size_", size, "_time_3600/run_0/failed_stats.txt",sep=""), sep=":")[3, 2]
        
        if(app[j] == "MatMulGPU"){
            matmulgpu <- c(gtx , k20, k40)
        }
        
        if(app[j] == "MatMulUn"){
            matmulun <- c(gtx , k20, k40)
        }
        if(app[j] == "MatMulShared"){
            matmulshared <- c(gtx , k20, k40)
        }            
        if(app[j] == "MatMulSharedUn"){
            matmulsharedun <- c(gtx , k20, k40)
        }
        if (app[j] == "SubSeqMax"){
            subseqmax <- c(gtx , k20, k40)
        }
        
        if (app[j] == "Bitonic"){
            Bitonic <- c(gtx , k20, k40)
        }
        if (app[j] == "Quicksort"){
            QuickSort <- c(gtx , k20, k40)
        }
        if (app[j] == "Quicksort"){
            VecAdd <- c(gtx , k20, k40)
        } 
        
    }
    
    
    
    print(matmulun)
    print(matmulgpu)
    print(matmulshared)
    print(matmulsharedun)
    print(QuickSort)
    print(VecAdd)
    print(Bitonic)
    print(subseqmax)
    
    final <- data.frame(MMU=matmulun, MMG=matmulgpu, MMSU=matmulsharedun, MMS=matmulshared, SSM=subseqmax, VAdd=VecAdd, Bitonic=Bitonic, QuickS=QuickSort)
    
    setEPS()
    postscript(paste("../images/SummaryFailed.eps",sep=""),
               height = 10, width = 18)
    par(mar=c(4, 9, 1, 1) + 0.1, mgp=c(7, 1.5, 0), las=1)
    
    barplot(as.matrix(final),
            ylab="Percentage of Failed Configurations",
            beside=T,
            ylim=c(0, .6),
            xpd=F,
            col=gray.colors(3, start=0, end=1),
            cex.names = 3,
            space=c(0,0.3),
            names=c("#1", "#2", "#3", "#4", "SSM", "VAdd", "Bit", "QSort"),
            cex.axis = 3,
            cex.lab = 3
    )
    legend("topright", c("GTX-980", "GTX-750", "Tesla-K40"), fill=gray.colors(3, start=0, end=1), cex=4)
    dev.off()
}

setwd(paste(dirpath, sep=""))
results_summary()
