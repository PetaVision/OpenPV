#include <OpenCL/opencl.h>

#define PV_USE_OPENCL
#define PV_CL_IMPL

#include "../src/columns/HyPerCol.hpp"
#include "../src/layers/LIF.hpp"
#include "../src/arch/opencl/CLDevice.hpp"
#include "../src/arch/opencl/CLKernel.hpp"
#include "../src/arch/opencl/CLBuffer.hpp"
#include "../src/utils/Timer.hpp"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define DEVICE 1

#define NX 256
#define NY 256
#define NXL 16
#define NYL 8

int check_results(uint4 * rnd_state, uint4 * rnd_state2, int count);
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
   
   PV::HyPerCol * hc = new PV::HyPerCol("test_cl_lif column", argc, argv);
   PV::LIF      * l1 = new PV::LIF("test_cl_lif layer", hc);

   exit(1);

   // query and print information about the devices found
   //
   if (query) cld->query_device_info();
   
   if (device == 0) {
      nxl = NXL;  nyl = NYL;
   }
   else {
      nxl = 1;  nyl = 1;
   }

   kernel = cld->createKernel("kernels/LIF_update_state.cl", "LIF_update_state",
                "-I /Users/rasmussn/eclipse/workspace.petavision/PetaVisionII/src/kernels/");
   
   exit(0);

   const size_t mem_size = NX*NY*sizeof(uint4);
   
   // create device buffer
   //
//   PV::CLBuffer * d_rnd_state = cld->createBuffer(CL_MEM_COPY_HOST_PTR, mem_size, rnd_state);

   // time running kernel locally
   //

   status |= kernel->setKernelArg(argid++, NX);
   status |= kernel->setKernelArg(argid++, NY);
   status |= kernel->setKernelArg(argid++, 1);
   status |= kernel->setKernelArg(argid++, 0);
//   status |= kernel->setKernelArg(argid++, d_rnd_state);

   printf("Executing on device...");
   kernel->run(NX*NY, nxl*nyl);
   
   // Check results for accuracy
   //
   printf("\nChecking results...");
//   d_rnd_state->copyFromDevice();
//   status = check_results(rnd_state, rnd_state2, NX*NY);
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
//   for (int n = 0; n < nLoops; n++) {
//      c_rand(rnd_state2, NX*NY);
//   }
   timer.stop();
   timer.elapsed_time();

   // Shutdown and cleanup
   //
   //   clReleaseMemObject(d_rnd_state);
   delete cld;
   
   printf("Finished...\n");
   
   return status;
}

#ifdef NOTYET
int check_results(uint4 * rnd_state, uint4 * rnd_state2, int count)
{
   uint4 state;

   for (int k = 0; k < count; k++) {
      state = cl_random_state(rnd_state2[k]);
      if (state.s0 != rnd_state[k].s0) {
         printf("check_results: results differ at k==%d, rnd_state==%d, rnd_state2==%d\n",
                k, state.s0, rnd_state[k].s0);
         return 1;
      }
   }
   return 0;
}

void c_rand(uint4 * rnd_state, int count)
{
   for (int k = 0; k < count; k++) {
      rnd_state[k] = cl_random_state(rnd_state[k]);
   }
}
#endif
