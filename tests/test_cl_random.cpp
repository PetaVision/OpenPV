#undef PV_USE_OPENCL

#include "../src/arch/opencl/CLDevice.hpp"
#include "../src/arch/opencl/CLKernel.hpp"
#include "../src/arch/opencl/CLBuffer.hpp"
#include "../src/arch/opencl/pv_uint4.h"
#include "../src/utils/Timer.hpp"
#include "../src/utils/cl_random.h"

#include "../src/kernels/cl_random.hcl"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#undef DEBUG_OUTPUT

#define DEVICE 1

#define NX 256
#define NY 256
#define NXL 16
#define NYL 8

int check_results(uint4 * rnd_state, uint4 * rnd_state2, int count);
int print_results(uint4 * rnd_state, uint4 * rnd_state2, int count);

void c_rand(uint4 * rnd_state, int count);

int main(int argc, char * argv[])
{
   unsigned int nxl, nyl;

   PV::Timer timer;
   
   int argid  = 0;
   int query  = 0;
   int device = DEVICE;
   int status = CL_SUCCESS;
   
   int nWarm = 10, nLoops = 100;

   if (argc > 1) {
      device = atoi(argv[1]);
   }
	
   PV::CLKernel * kernel;
   PV::CLDevice * cld = new PV::CLDevice(device);
   
   // query and print information about the devices found
   //
   if (query) cld->query_device_info();
   
   if (device == 0) {
      nxl = NXL;
      nyl = NYL;
   }
   else {
      nxl = 1;
      nyl = 1;
   }

   kernel = cld->createKernel("kernels/test_cl_random.cl", "cl_rand");
   
   const size_t mem_size = NX*NY*sizeof(uint4);
   
   uint4 * rnd_state  = cl_random_init(NX*NY);
   uint4 * rnd_state2 = cl_random_init(NX*NY);

   assert(rnd_state  != NULL);
   assert(rnd_state2 != NULL);
   
   // create device buffer
   //
   PV::CLBuffer * d_rnd_state = cld->createBuffer(CL_MEM_COPY_HOST_PTR, mem_size, rnd_state);

   status |= kernel->setKernelArg(argid++, NX);
   status |= kernel->setKernelArg(argid++, NY);
   status |= kernel->setKernelArg(argid++, 1);
   status |= kernel->setKernelArg(argid++, 0);
   status |= kernel->setKernelArg(argid++, d_rnd_state);

   printf("Executing on device...");
   kernel->run(NX*NY, nxl*nyl);
   printf("\n");
   
#ifdef DEBUG_OUTPUT
   status = print_results(rnd_state, rnd_state2, NX*NY);
#endif

   // Check results for accuracy
   //
   printf("Checking results...");
   d_rnd_state->copyFromDevice();
   status = check_results(rnd_state, rnd_state2, NX*NY);
   if (status != CL_SUCCESS) {
      exit(status);
   }

   printf(" CORRECT\n");

   for (int n = 0; n < nWarm; n++) {
      kernel->run(NX*NY, nxl*nyl);
   }

   printf("Timing %d loops on device... ", nLoops);
   timer.start();
   for (int n = 0; n < nLoops; n++) {
      kernel->run(NX*NY, nxl*nyl);
   }
   status |= kernel->finish();
   timer.stop();
   timer.elapsed_time();

   printf("Timing %d loops on host..... ", nLoops);
   timer.start();
   for (int n = 0; n < nLoops; n++) {
      c_rand(rnd_state2, NX*NY);
   }
   timer.stop();
   timer.elapsed_time();

   // Shutdown and cleanup
   //
   //   clReleaseMemObject(d_rnd_state);
   delete cld;
   
   printf("Finished...\n");
   
   return status;
}

int check_results(uint4 * rnd_state, uint4 * rnd_state2, int count)
{
   uint4 state;

   for (int k = 0; k < count; k++) {
      state = cl_random_get(rnd_state2[k]);
      if (state.s0 != rnd_state[k].s0) {
         printf("check_results: results differ at k==%d, rnd_state==%d, rnd_state2==%d\n",
                k, state.s0, rnd_state[k].s0);
         return 1;
      }
      float p1 = (float) ((float)  state.s0 / (float)  4294967296.0);
      float p2 = (float) ((double) state.s0 / (double) 4294967296.0);
      if (p1 != p2) printf("check_results[%d]: %f %f\n", k, p1, p2);
   }
   return 0;
}

int print_results(uint4 * rnd_state, uint4 * rnd_state2, int count)
{
   uint4 state;

   FILE * fp = fopen("test_cl_random.results", "w");

   for (int k = 1; k < count; k+=2) {
      fprintf(fp, "%f %f   %f %f\n",
              cl_random_prob(rnd_state[ k-1]), cl_random_prob(rnd_state[ k]),
              cl_random_prob(rnd_state2[k-1]), cl_random_prob(rnd_state2[k])
              );
   }
   fclose(fp);
   return 0;
}

void c_rand(uint4 * rnd_state, int count)
{
   for (int k = 0; k < count; k++) {
      rnd_state[k] = cl_random_get(rnd_state[k]);
   }
}

