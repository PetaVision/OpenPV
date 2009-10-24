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

cl_info: cl_info.o CLDevice.o
	$(CPP) -o cl_info $(LDFLAGS) $< CLDevice.o

clean:
	rm -f *.o cl_info
