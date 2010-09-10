//============================================================================
// Name        : gpu.cpp
// Author      : Craig Rasmussen
// Description : main driver for testing GPU kernels
//============================================================================

#include "src/utils/Timer.hpp"
#include "src/arch/opencl/CLDevice.hpp"
#include "src/arch/opencl/CLBuffer.hpp"
#include "src/arch/opencl/CLKernel.hpp"

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

   pvdata_t * G_E = (pvdata_t *) malloc(numNeurons * sizeof(pvdata_t));
   assert(G_E != NULL);

   pvdata_t * phiExc = (pvdata_t *) malloc(numNeurons * sizeof(pvdata_t));
   assert(phiExc != NULL);

   pvdata_t * G_I = (pvdata_t *) malloc(numNeurons * sizeof(pvdata_t));
   assert(G_I != NULL);

   pvdata_t * phiInh = (pvdata_t *) malloc(numNeurons * sizeof(pvdata_t));
   assert(phiInh != NULL);

   pvdata_t * G_IB = (pvdata_t *) malloc(numNeurons * sizeof(pvdata_t));
   assert(G_IB != NULL);

   pvdata_t * phiInhB = (pvdata_t *) malloc(numNeurons * sizeof(pvdata_t));
   assert(phiInhB != NULL);


   // warm up memory
   timer.start();
   bzero(V, numNeurons*sizeof(pvdata_t));
   bzero(G_E, numNeurons*sizeof(pvdata_t));
   bzero(phiExc, numNeurons*sizeof(pvdata_t));
   bzero(G_I, numNeurons*sizeof(pvdata_t));
   bzero(phiInh, numNeurons*sizeof(pvdata_t));
   bzero(G_IB, numNeurons*sizeof(pvdata_t));
   bzero(phiInhB, numNeurons*sizeof(pvdata_t));

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
   CLBuffer * d_G_E = cld->createBuffer(numNeurons*sizeof(pvdata_t), G_E);
   CLBuffer * d_phiExc = cld->createBuffer(numNeurons*sizeof(pvdata_t), phiExc);
   CLBuffer * d_G_I = cld->createBuffer(numNeurons*sizeof(pvdata_t), G_I);
   CLBuffer * d_phiInh = cld->createBuffer(numNeurons*sizeof(pvdata_t), phiInh);
   CLBuffer * d_G_IB = cld->createBuffer(numNeurons*sizeof(pvdata_t), G_IB);
   CLBuffer * d_phiInhB = cld->createBuffer(numNeurons*sizeof(pvdata_t), phiInhB);

   timer.start();
   pvdata_t * h_V = (pvdata_t*) d_V->map(CL_MAP_WRITE);
   pvdata_t * h_G_E = (pvdata_t*) d_G_E->map(CL_MAP_WRITE);
   pvdata_t * h_phiExc = (pvdata_t*) d_phiExc->map(CL_MAP_WRITE);
   pvdata_t * h_G_I = (pvdata_t*) d_G_I->map(CL_MAP_WRITE);
   pvdata_t * h_phiInh = (pvdata_t*) d_phiInh->map(CL_MAP_WRITE);
   pvdata_t * h_G_IB = (pvdata_t*) d_G_IB->map(CL_MAP_WRITE);
   pvdata_t * h_phiInhB = (pvdata_t*) d_phiInhB->map(CL_MAP_WRITE);


   for (int k = 0; k < numNeurons; k++) {
      h_V[k] = 1;
      h_G_E[k] = 1;
      h_phiExc[k] = 1;
      h_G_I[k] = 1;
      h_phiInh[k] = 1;
      h_G_IB[k] = 1;
      h_phiInhB[k] = 1;
   }

   d_V->unmap(h_V);
   d_G_E->unmap(h_G_E);
   d_phiExc->unmap(h_phiExc);
   d_G_I->unmap(h_G_I);
   d_phiInh->unmap(h_phiInh);
   d_G_IB->unmap(h_G_IB);
   d_phiInhB->unmap(h_phiInhB);

   timer.stop();
   printf("Init and map:         "); timer.elapsed_time();

   status = CL_SUCCESS;
   status |= kernel->addKernelArg(argid++, d_V);
   status != kernel->addKernelArg(argid++, d_G_E);
   status != kernel->addKernelArg(argid++, d_phiExc);
   status != kernel->addKernelArg(argid++, d_G_I);
   status != kernel->addKernelArg(argid++, d_phiInh);
   status != kernel->addKernelArg(argid++, d_G_IB);
   status != kernel->addKernelArg(argid++, d_phiInhB);

   status |= kernel->addKernelArg(argid++, nx);
   status |= kernel->addKernelArg(argid++, ny);
   status |= kernel->addKernelArg(argid++, nPad);

   float d_elapsed = 0.0f;
   timer.start();
   for (int n = 0; n < 1; n++) {
      kernel->run(nx, ny, nxl, nyl);
      d_elapsed += ((float) kernel->get_execution_time()) / 1.0e6;
   }
   timer.stop();
   printf("Executing on device:  "); timer.elapsed_time();
   printf("Elapsed time on device:             device time == %f \n", d_elapsed);

   // examine results
   h_V = (pvdata_t*) d_V->map(CL_MAP_READ);
   h_G_E = (pvdata_t*) d_G_E->map(CL_MAP_READ);
   h_phiExc = (pvdata_t*) d_phiExc->map(CL_MAP_READ);
   h_G_I = (pvdata_t*) d_G_I->map(CL_MAP_READ);
   h_phiInh = (pvdata_t*) d_phiInh->map(CL_MAP_READ);
   h_G_IB = (pvdata_t*) d_G_IB->map(CL_MAP_READ);
   h_phiInhB = (pvdata_t*) d_phiInhB->map(CL_MAP_READ);

   for (int k = 400; k < 500; k++){
   //printf("V[0:1] == %f %f\n", h_V[0], h_V[1]);
   printf("G_E[k] == %d %f\n", k, h_G_E[k]);
   //printf("phiExc[0:1] == %f %f\n", h_phiExc[0], h_phiExc[1]);
   printf("G_I[k] == %d %f\n", k, h_G_I[k]);
   //printf("phiInh[0:1] == %f %f\n", h_phiInh[0], h_phiInh[1]);
   printf("G_IB[k] == %d %f\n", k, h_G_IB[k]);
   printf("\n");
   //printf("phiInhB[0:1] == %f %f\n", h_phiInhB[0], h_phiInhB[1]);
   }


   d_V->unmap(h_V);
   d_G_E->unmap(h_G_E);
   d_phiExc->unmap(h_phiExc);
   d_G_I->unmap(h_G_I);
   d_phiInh->unmap(h_phiInh);
   d_G_IB->unmap(h_G_IB);
   d_phiInhB->unmap(h_phiInhB);

// Shutdown and cleanup
   //
   delete d_V;
   delete d_G_E;
   delete d_phiExc;
   delete d_G_I;
   delete d_phiInh;
   delete d_G_IB;
   delete d_phiInhB;

   delete cld;

   printf("Finished...\n");


   return status;
}
