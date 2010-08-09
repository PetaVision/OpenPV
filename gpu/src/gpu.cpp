//============================================================================
// Name        : gpu.cpp
// Author      : Craig Rasmussen
// Description : main driver for testing GPU kernels
//============================================================================

#include "src/utils/Timer.hpp"
#include "src/arch/opencl/CLDevice.hpp"
#include "src/arch/opencl/CLBuffer.hpp"
#include "src/arch/opencl/CLKernel.hpp"
//#include "src/arch/opencl/CLBuffer.hpp"

#include <stdio.h>
#include <stdlib.h>

#define CONVOLVE
#define DEBUG_OUTPUT

#define NXL       16
#define NYL       16
#define NXGLOBAL  4096
#define NYGLOBAL  4096

#define NPAD      4
#define NPAD2     (2*NPAD)

#define EXT_SIZE  ( (NXGLOBAL+NPAD2) * (NYGLOBAL+NPAD2) )
#define IMG_SIZE  (  NXGLOBAL * NYGLOBAL  )

#define pvdata_t cl_float

using namespace PV;

int main(int argc, char * argv[])
{
   uint nxl, nyl;

   Timer timer;
   CLKernel * kernel;

   int argid  = 0;
   int query  = 0;
   int device = 1;
   int status = CL_SUCCESS;

   if (argc > 1) {
      device = atoi(argv[1]);
   }

   const unsigned int nx = NXGLOBAL;
   const unsigned int ny = NYGLOBAL;
   const unsigned int nPad = NPAD;

   const unsigned int numNeurons = nx * ny;

   const unsigned int sx = 1;
   const unsigned int sy = nx + 2*nPad;

   pvdata_t * V = (pvdata_t *) malloc(numNeurons * sizeof(pvdata_t));
   assert(V != NULL);

   // warm up memory
   timer.start();
   bzero(V, numNeurons*sizeof(pvdata_t));
   timer.stop();
   printf("Initializing on host: "); timer.elapsed_time();

   CLDevice * cld = new CLDevice(device);

   // query and print information about the devices found
   //
   if (query) cld->query_device_info();

   if (device == 0) {
      nxl = NXL;
      nyl = NYL;
      kernel = cld->createKernel("mem.cl", "bandwidth");
   }
   else {
      nxl = 1;
      nyl = 1;
      kernel = cld->createKernel("mem.cl", "bandwidth");
   }

   // create memory buffers
   //

   CLBuffer * d_V = cld->createBuffer(numNeurons*sizeof(pvdata_t), V);

   timer.start();
   pvdata_t * h_V = (pvdata_t*) d_V->map(CL_MAP_WRITE);
   for (int k = 0; k < numNeurons; k++) {
      h_V[k] = 1;
   }
   d_V->unmap(h_V);
   timer.stop();
   printf("Init and map:         "); timer.elapsed_time();

   status = CL_SUCCESS;
   status |= kernel->addKernelArg(argid++, d_V);
   status |= kernel->addKernelArg(argid++, nx);
   status |= kernel->addKernelArg(argid++, ny);
   status |= kernel->addKernelArg(argid++, nPad);

   float d_elapsed = 0.0f;
   timer.start();
   for (int n = 0; n < 100; n++) {
      kernel->run(nx, ny, nxl, nyl);
      d_elapsed += ((float) kernel->get_execution_time()) / 1.0e6;
   }
   timer.stop();
   printf("Executing on device:  "); timer.elapsed_time();
   printf("Elapsed time on device:             device time == %f \n", d_elapsed);

   // examine results
   h_V = (pvdata_t*) d_V->map(CL_MAP_READ);
   printf("V[0:1] == %f %f\n", h_V[0], h_V[1]);
   d_V->unmap(h_V);

// Shutdown and cleanup
   //
   delete d_V;
   delete cld;

   printf("Finished...\n");


   return status;
}
