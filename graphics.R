dirpath <- "./experiments/"

#General Configurations of the graphics
cexTam=1.25
PaletteColor <- c("red", "blue", "darkgray", "orange","black","lightblue", "lightblue","violet")

graphics <- function(){
    setEPS()
    postscript(paste("../../../images/", app[j], "-", sizes[k], "-", gpu[i],
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
    
    postscript(paste("../../../images/", app[j],"-", sizes[k], "-", gpu[i],
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

calc_speedup <- function(old, new){
    return (median(old[old < 9999]) / median(new[new < 9999]))
}

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
        
        
        #         1048576
        gtx   <- scan(paste("./GTX-980/", app[j], "/size_", size, "_time_3600/run_0/benchmark.txt",sep=""))
        k20   <- scan(paste("./GTX-750/", app[j], "/size_", size, "_time_3600/run_0/benchmark.txt",sep=""))
        k40   <- scan(paste("./Tesla-K40/", app[j], "/size_", size, "_time_3600/run_0/benchmark.txt",sep=""))
        o2gtx <- scan(paste("./GTX-980/", app[j], "/size_", size, "_baseline/opt_2.txt",sep=""))
        o2k20 <- scan(paste("./GTX-750/", app[j], "/size_", size, "_baseline/opt_2.txt",sep=""))
        o2k40 <- scan(paste("./Tesla-K40/", app[j], "/size_", size, "_baseline/opt_2.txt",sep=""))            
        
        print("MMG")
        print("GTX")
        print(calc_speedup(o2gtx, gtx))
        print("K20")
        print(calc_speedup(o2k20, k20))
        print("K40") 
        print(calc_speedup(o2k40, k40))
        
        if(app[j] == "MatMulGPU"){
            matmulgpu <- c(calc_speedup(o2gtx, gtx), calc_speedup(o2k20, k20), calc_speedup(o2k40, k40))
        }
        
        if(app[j] == "MatMulUn"){
            matmulun <- c(calc_speedup(o2gtx, gtx), calc_speedup(o2k20, k20), calc_speedup(o2k40, k40))
        }
        if(app[j] == "MatMulShared"){
            matmulshared <- c(calc_speedup(o2gtx, gtx), calc_speedup(o2k20, k20), calc_speedup(o2k40, k40))
        }            
        if(app[j] == "MatMulSharedUn"){
            matmulsharedun <- c(calc_speedup(o2gtx, gtx), 0, calc_speedup(o2k40, k40))
        }
        if (app[j] == "SubSeqMax"){
            subseqmax <- c(calc_speedup(o2gtx, gtx), calc_speedup(o2k20, k20), calc_speedup(o2k40, k40))
        }
        
        if (app[j] == "Bitonic"){
            Bitonic <- c(calc_speedup(o2gtx, gtx), calc_speedup(o2k20, k20), calc_speedup(o2k40, k40))
        }
        if (app[j] == "Quicksort"){
            QuickSort <- c(calc_speedup(o2gtx, gtx), calc_speedup(o2k20, k20), calc_speedup(o2k40, k40))
        }
        if (app[j] == "Quicksort"){
            VecAdd <- c(calc_speedup(o2gtx, gtx), calc_speedup(o2k20, k20), calc_speedup(o2k40, k40))
        }         
    }
    
    final <- data.frame(MMU=matmulun, MMG=matmulgpu, MMSU=matmulsharedun, MMS=matmulshared, SSM=subseqmax, VAdd=VecAdd, Bitonic=Bitonic, QuickS=QuickSort)
    
    setEPS()
    postscript(paste("../images/Summary.eps",sep=""),
               height = 10, width = 18)
    par(mar=c(4, 9, 1, 1) + 0.1, mgp=c(7, 1.5, 0), las=1)
    
    barplot(as.matrix(final),
            ylab="Percentage of Speedup vs. -O2",
            beside=T,
            ylim=c(1, 8),
            xpd=F,
            col=gray.colors(3, start=0, end=1),
            cex.names = 3,
            space=c(0,0.3),
            names=c("#1", "#2", "#3", "#4", "SSM", "VAdd", "Bit", "QSort"),
            cex.axis = 3,
            cex.lab = 3
    )
    legend("topright", c("GTX-980", "GTX-70", "Tesla-K40"), fill=gray.colors(3, start=0, end=1), cex=4)
    dev.off()
}

setwd(paste(dirpath, sep=""))
results_summary()
    
    gpu <- c("Tesla-K40", "GTX-750", "GTX-980")

    for(i in 1:length(gpu)){
        print(getwd())
        setwd(paste(gpu[i], sep=""))
        print(getwd())
        
#         if (gpu[i]== "GTX-680"){
#             app <- c("MatMulGPU", "MatMulShared", "MatMulSharedUn", "MatMulUn", "SubSeqMax")
#         }
#         
#         if (gpu[i]== "Tesla-K20"){
#             app <- c("MatMulGPU", "MatMulShared",  "MatMulUn", "SubSeqMax")
#         }
        if (gpu[i]== "Tesla-K40"){
            app <- c("MatMulGPU", "MatMulShared", "MatMulSharedUn", "MatMulUn", 
                     "SubSeqMax", "Bitonic", "Quicksort", "VecAdd","backprop", 
                     "gaussian", "hotspot", "kmeans", "lud", "nn", 
                     "bfs", "b+tree", "heartwall", "hybridsort", "lavaMD", "myocyte")
        }    
        if (gpu[i]== "GTX-980" | gpu[i]== "GTX-750"){
            app <- c("MatMulGPU", "MatMulShared", "MatMulSharedUn", "MatMulUn",
                     "SubSeqMax", "Bitonic", "Quicksort", "VecAdd","backprop",
                     "gaussian", "hotspot", "lud", "bfs", "b+tree", "heartwall",
                     "lavaMD", "myocyte")
        }    

                
        for(j in 1:length(app)){
            print(getwd())
            setwd(paste("./", app[j], sep=""))
            print(getwd())
            if (app[j] == "MatMulGPU" | app[j] == "MatMulShared" | app[j] == "MatMulUn" | app[j] == "MatMulSharedUn"){
                sizes <- c(2048, 4096, 8192, 16384, 32768)
                for(k in 1:length(sizes)){
                    opt0 <- scan(paste("./size_", sizes[k], "_baseline/opt_0.txt",sep=""))
                    opt1 <- scan(paste("./size_", sizes[k], "_baseline/opt_1.txt",sep=""))
                    opt2 <- scan(paste("./size_", sizes[k], "_baseline/opt_2.txt",sep=""))
                    opt3 <- scan(paste("./size_", sizes[k], "_baseline/opt_3.txt",sep=""))
                    benchmark <- scan(paste("./size_", sizes[k], "_time_3600/run_0/benchmark.txt",sep=""))
                    logAll <- read.table(paste("./size_", sizes[k], "_time_3600/run_0/logall.txt",sep=""))
                    logBest <- read.table(paste("./size_", sizes[k], "_time_3600/run_0/logbest.txt",sep=""))

                    graphics()
                }
            }

            if (app[j] == "SubSeqMax"){
                sizes <- c(67108864, 134217728, 268435456, 536870912, 1073741824 )
                for(k in 1:length(sizes)){
                    opt0 <- scan(paste("./size_", sizes[k], "_baseline/opt_0.txt",sep=""))
                    opt1 <- scan(paste("./size_", sizes[k], "_baseline/opt_1.txt",sep=""))
                    opt2 <- scan(paste("./size_", sizes[k], "_baseline/opt_2.txt",sep=""))
                    opt3 <- scan(paste("./size_", sizes[k], "_baseline/opt_3.txt",sep=""))
                    benchmark <- scan(paste("./size_", sizes[k], "_time_3600/run_0/benchmark.txt",sep=""))
                    logAll <- read.table(paste("./size_", sizes[k], "_time_3600/run_0/logall.txt",sep=""))
                    logBest <- read.table(paste("./size_", sizes[k], "_time_3600/run_0/logbest.txt",sep=""))
                    
                    graphics()
                }
            }            
            if (app[j] == "Bitonic" ){
                sizes <- c(262144, 524288, 1048576, 2097152, 4194304 )
                for(k in 1:length(sizes)){
                    opt0 <- scan(paste("./size_", sizes[k], "_baseline/opt_0.txt",sep=""))
                    opt1 <- scan(paste("./size_", sizes[k], "_baseline/opt_1.txt",sep=""))
                    opt2 <- scan(paste("./size_", sizes[k], "_baseline/opt_2.txt",sep=""))
                    opt3 <- scan(paste("./size_", sizes[k], "_baseline/opt_3.txt",sep=""))
                    benchmark <- scan(paste("./size_", sizes[k], "_time_3600/run_0/benchmark.txt",sep=""))
                    logAll <- read.table(paste("./size_", sizes[k], "_time_3600/run_0/logall.txt",sep=""))
                    logBest <- read.table(paste("./size_", sizes[k], "_time_3600/run_0/logbest.txt",sep=""))
                    
                    graphics()
                }
            }
            if (app[j] == "Quicksort"){
                sizes <- c(4096, 8192, 16384, 32768, 65536 )
                for(k in 1:length(sizes)){
                    opt0 <- scan(paste("./size_", sizes[k], "_baseline/opt_0.txt",sep=""))
                    opt1 <- scan(paste("./size_", sizes[k], "_baseline/opt_1.txt",sep=""))
                    opt2 <- scan(paste("./size_", sizes[k], "_baseline/opt_2.txt",sep=""))
                    opt3 <- scan(paste("./size_", sizes[k], "_baseline/opt_3.txt",sep=""))
                    benchmark <- scan(paste("./size_", sizes[k], "_time_3600/run_0/benchmark.txt",sep=""))
                    logAll <- read.table(paste("./size_", sizes[k], "_time_3600/run_0/logall.txt",sep=""))
                    logBest <- read.table(paste("./size_", sizes[k], "_time_3600/run_0/logbest.txt",sep=""))
                    
                    graphics()
                }
            }
            
            if (app[j] == "VecAdd"){
                sizes <- c(32768, 131072, 262144, 1048576, 4194304 )
                for(k in 1:length(sizes)){
                    opt0 <- scan(paste("./size_", sizes[k], "_baseline/opt_0.txt",sep=""))
                    opt1 <- scan(paste("./size_", sizes[k], "_baseline/opt_1.txt",sep=""))
                    opt2 <- scan(paste("./size_", sizes[k], "_baseline/opt_2.txt",sep=""))
                    opt3 <- scan(paste("./size_", sizes[k], "_baseline/opt_3.txt",sep=""))
                    benchmark <- scan(paste("./size_", sizes[k], "_time_3600/run_0/benchmark.txt",sep=""))
                    logAll <- read.table(paste("./size_", sizes[k], "_time_3600/run_0/logall.txt",sep=""))
                    logBest <- read.table(paste("./size_", sizes[k], "_time_3600/run_0/logbest.txt",sep=""))
                    
                    graphics()
                }
            }
            
            if (app[j] == "ParticleFilterNaive" & gpu[i] == "Tesla-K40"){
                opt0 <- scan(paste("./size_50000_baseline/opt_0.txt",sep=""))
                opt1 <- scan(paste("./size_50000_baseline/opt_1.txt",sep=""))
                opt2 <- scan(paste("./size_50000_baseline/opt_2.txt",sep=""))
                opt3 <- scan(paste("./size_50000_baseline/opt_3.txt",sep=""))
                benchmark <- scan(paste("./size_50000_time_3600/run_0/benchmark.txt",sep=""))
                logAll <- read.table(paste("./size_50000_time_3600/run_0/logall.txt",sep=""))
                logBest <- read.table(paste("./size_50000_time_3600/run_0/logbest.txt",sep=""))
                graphics()
            }
            
            if (app[j] == "backprop" | app[j] ==  "gaussian" | app[j] ==  "hotspot" | 
                    app[j] ==  "kmeans" | app[j] ==  "lud" | app[j] ==  "nn" | 
                    app[j] ==  "bfs" | app[j] ==  "b+tree" | app[j] ==  "heartwall" | 
                    app[j] ==  "hybridsort" | app[j] ==  "lavaMD" | app[j] ==  "myocyte"){
                sizes <- c(0)
                opt0 <- scan(paste("./size_default_baseline/opt_0.txt",sep=""))
                opt1 <- scan(paste("./size_default_baseline/opt_1.txt",sep=""))
                opt2 <- scan(paste("./size_default_baseline/opt_2.txt",sep=""))
                opt3 <- scan(paste("./size_default_baseline/opt_3.txt",sep=""))
                benchmark <- scan(paste("./size_default_time_3600/run_0/benchmark.txt",sep=""))
                logAll <- read.table(paste("./size_default_time_3600/run_0/logall.txt",sep=""))
                logBest <- read.table(paste("./size_default_time_3600/run_0/logbest.txt",sep=""))
                graphics()
            }

            print(getwd())
            setwd("..")
            print(getwd())
        }
        setwd("..")
    }
