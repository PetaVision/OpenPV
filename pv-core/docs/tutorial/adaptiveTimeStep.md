===================
# Adaptive timescale
===================

This article is about the HyPerCol parameter dtAdaptFlag and the related parameters in HyPercol and ANNNormalizedErrorLayer.

The motivation for an adaptive timestep comes from the desire to solve the differential equation of the ANNNormalizedErrorLayer as efficiently and accurately as possible. When we have a large error, we'll want to reduce our timestep; when we have a small error, we can afford to take a larger timestep. 

[graphic comparing run-time and timers with and without adaptive timestep]

## Parameters
The following are the key HyPerCol parameters you'll need to set in order to use the adaptive timestep

  Parameter                | Typical Value |   Description
---------------------------|---------------|------------------------------------------------------
  dtAdaptFlag              |  True; False; | Declares if you are using the Adaptive Timestep
  dtScaleMax               |  1 - 10       | Specifies the maximum timescale allowed
  dtScaleMin               |  0.1 - 1      | Sets the default timescale; timescale can drop below if error high
  dtChangeMax              |  0.01 - 0.2   | Maximum % change in error dt will adapt to 
  dtChangeMin              |  0 - 0.001    | Minimum % change in error dt will adapt to
  dtMinToleratedTimeScale  |  0.0001       | If the timescale drops below this value, the run will abort
  
* If the % change is outside of the range of dtScaleMax and dtScaleMin, the timescale will reset to the dtScaleMin
 
Review the doxygen documentation for the HyPerCol Parameters for more details:
http://petavision.sourceforge.net/doxygen/html/classPV_1_1HyPerCol.html#member-group

## Troublshooting



goal: 
1) avoid / recover from instabilities
	- runs used to blow up from numerical instabilities / take forever
	-

dt = dt/tau 		
	- really adjusting tau, 
	
if error high:
	increase tau to max default value
	

if error low:


adjusting rate of exp


tau never changes,  
	dt that the layer is told to used
	
dt/tau	

argument dt that gets passed to update state, 

sending update state dt/tau

updateState(t, dt) 		where dt = dt/tau

dt = .5		tau = 10
dt = 1		tau = 20

dt is determined by the adapt timescale stuff at the hypercol level
	- what's the largest the largest you can afford
	- chooses the lowest of all the layers

numerical instabilities occur, b/c you have a rate of change 
and you are going to 
if the derivative is too big, you can over shoot .. 
