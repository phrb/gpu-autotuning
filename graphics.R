dirpath <- "/home/marcos/Dropbox/Doctorate/Results/gpu-autotuning/experiments/"

#General Configurations of the graphics
cexTam=1.25
PaletteColor <- c("red", "blue", "darkgray", "orange","black","lightblue", "lightblue","violet")

#TODO
#Graficos de lineas dos LogBest Para ver cuantas vezes de 3600 são necesarias para encontrar um minimo máximo

graphics <- function(){
  setEPS()
  postscript(paste("../../../images/", app[j], gpu[i], "-Box.eps",sep=""))
  boxplot(opt0[(opt0 < 9999)], opt1[(opt1 < 9999)], opt2[(opt2 < 9999)], 
          opt3[(opt3 < 9999)], logAll[(logAll<9999)], logBest[(logBest < 9999)],
          names = c("Opt 0", "Opt 1","Opt 2","Opt 3", "All", "OTuner"),
          main=paste("Opentuner of NVCC of ", app[j], "over ", gpu[i],sep=""),
          ylab="Time in Miliseconds", 
          cex.main=1.45,
          cex.axis = 1.45, 
          cex.lab = 1.35       
  )
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
    
            if (app[j] == "MatMulGPU" | app[j] == "MatMulShared" | app[j] == "./MatMulUn"){
                opt0 <- scan(paste("./size_1024_baseline/opt_0.txt",sep=""))                         
                opt1 <- scan(paste("./size_1024_baseline/opt_1.txt",sep=""))
                opt2 <- scan(paste("./size_1024_baseline/opt_2.txt",sep=""))
                opt3 <- scan(paste("./size_1024_baseline/opt_3.txt",sep=""))
                logAll <- read.table(paste("./size_1024_time_3600/run_0/logall.txt",sep=""))[["V2"]]
                logBest <- read.table(paste("./size_1024_time_3600/run_0/logbest.txt",sep=""))[["V2"]]
                
                graphics()
             
            }
            
            if (app[j] == "MatMulSharedUn" & gpu[i] == "Tesla-K20"){
                opt0 <- scan(paste("./size_256_baseline/opt_0.txt",sep=""))                         
                opt1 <- scan(paste("./size_256_baseline/opt_1.txt",sep=""))
                opt2 <- scan(paste("./size_256_baseline/opt_2.txt",sep=""))
                opt3 <- scan(paste("./size_256_baseline/opt_3.txt",sep=""))
                logAll <- read.table(paste("./size_256_time_3600/run_0/logall.txt",sep=""))[["V2"]]
                logBest <- read.table(paste("./size_256_time_3600/run_0/logbest.txt",sep=""))[["V2"]]
                
                graphics()
                
            }
    
    
          if (app[j] == "SubSeqMax" & gpu[i] == "Tesla-K40"){
            opt0 <- scan(paste("./size_134217728_baseline/opt_0.txt",sep=""))                         
            opt1 <- scan(paste("./size_134217728_baseline/opt_1.txt",sep=""))
            opt2 <- scan(paste("./size_134217728_baseline/opt_2.txt",sep=""))
            opt3 <- scan(paste("./size_134217728_baseline/opt_3.txt",sep=""))
            logAll <- read.table(paste("./size_134217728_time_3600/run_0/logall.txt",sep=""))[["V2"]]
            logBest <- read.table(paste("./size_134217728_time_3600/run_0/logbest.txt",sep=""))[["V2"]]            
            graphics()            
          }        
          
          if (app[j] == "SubSeqMax" & gpu[i] == "Tesla-K20"){
            opt0 <- scan(paste("./size_1073741824_baseline/opt_0.txt",sep=""))                         
            opt1 <- scan(paste("./size_1073741824_baseline/opt_1.txt",sep=""))
            opt2 <- scan(paste("./size_1073741824_baseline/opt_2.txt",sep=""))
            opt3 <- scan(paste("./size_1073741824_baseline/opt_3.txt",sep=""))
            logAll <- read.table(paste("./size_1073741824_time_3600/run_0/logall.txt",sep=""))[["V2"]]
            logBest <- read.table(paste("./size_1073741824_time_3600/run_0/logbest.txt",sep=""))[["V2"]]            
            graphics()            
          }        
#     
    
    
    
    setwd("../")
  }
}    
