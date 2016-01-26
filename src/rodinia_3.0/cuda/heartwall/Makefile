ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o 
	nvcc -ccbin gcc-4.8 main.o ./AVI/avilib.o ./AVI/avimod.o -I/usr/local/cuda/include -lm -o heartwall

# compile main function file into object (binary)
main.o: main.cu kernel.cu define.c
	nvcc -ccbin gcc-4.8 $(OUTPUT) $(KERNEL_DIM) main.cu $(NVCC_FLAGS) -I./AVI -c 

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo
