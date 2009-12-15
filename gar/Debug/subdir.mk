################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../RetinaGrating.cpp \
../gamma.cpp 

OBJS += \
./RetinaGrating.o \
./gamma.o 

CPP_DEPS += \
./RetinaGrating.d \
./gamma.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/Users/arick/Documents/workspace/PetaVision" -I"/Users/arick/Documents/workspace/PetaVision/src/arch/mpi" -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


