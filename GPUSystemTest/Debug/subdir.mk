################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../GPUTestForNinesProbe.cpp \
../GPUTestForOnesProbe.cpp \
../GPUTestForTwosProbe.cpp \
../GPUTestProbe.cpp \
../pv.cpp 

OBJS += \
./GPUTestForNinesProbe.o \
./GPUTestForOnesProbe.o \
./GPUTestForTwosProbe.o \
./GPUTestProbe.o \
./pv.o 

CPP_DEPS += \
./GPUTestForNinesProbe.d \
./GPUTestForOnesProbe.d \
./GPUTestForTwosProbe.d \
./GPUTestProbe.d \
./pv.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	mpic++ -O3 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


