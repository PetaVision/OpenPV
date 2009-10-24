CPP = g++

# Each subdirectory must supply rules for building sources it contributes
%.o: %.cpp
	@echo 'Building file: $@'
	@echo 'Invoking: GCC C++ Compiler'
	$(CPP) -c $<
	@echo 'Finished building: $@'
	@echo ' '
	
all: cl_info

cl_info: cl_info.o
	$(CPP) -o cl_info $<
