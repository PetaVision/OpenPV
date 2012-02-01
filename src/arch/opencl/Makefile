OPENCL_FRW = /System/Library/Frameworks/OpenCL.framework

CPP = g++

CPPFLAGS = -g -I$(OPENCL_FRW)/Headers
LDFLAGS = -framework OpenCL

# Each subdirectory must supply rules for building sources it contributes
%.o: %.cpp
	@echo 'Building file: $@'
	@echo 'Invoking: GCC C++ Compiler'
	$(CPP) -c $< $(CPPFLAGS)
	@echo 'Finished building: $@'
	@echo ' '
	
all: cl_info

conv: convolve_main.o CLDevice.o CLBuffer.o CLKernel.o ../../utils/Timer.o
	$(CPP) -o convolve_main $(LDFLAGS) $< CLBuffer.o CLKernel.o CLDevice.o Timer.o



cl_info: cl_info.o CLDevice.o CLBuffer.o CLKernel.o
	$(CPP) -o cl_info $(LDFLAGS) $< CLBuffer.o CLKernel.o CLDevice.o

clean:
	rm -f *.o cl_info
