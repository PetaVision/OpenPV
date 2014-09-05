#undef PV_USE_OPENCL

#include <columns/HyPerCol.hpp>
#include <layers/LIF.hpp>
#include <layers/Retina.hpp>
#include <arch/opencl/CLDevice.hpp>
#include <arch/opencl/CLKernel.hpp>
#include <arch/opencl/CLBuffer.hpp>
#include <utils/Timer.hpp>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// include the kernel for CPU calculations (without OpenCL)
//
#undef PV_USE_OPENCL
#include <kernels/LIF_update_state.cl>

namespace PV {

int test_kernels(int argc, char * argv[])
{
   Timer timer;
   CLKernel * kernel;
   CLKernel * retinaKernel;

   int status = CL_SUCCESS;
   int nWarm = 10, nLoops = 100;

   HyPerCol * hc = new HyPerCol("test_cl_all column", argc, argv, "..");
   Retina   * retina = new Retina("test_cl_all layer", hc);
   LIF      * l1 = new LIF("test_cl_all layer", hc);

   const PVLayerLoc * loc = l1->getLayerLoc();

   const int nx = loc->nx;
   const int ny = loc->ny;
   const int nf = loc->nf;
   const PVHalo * halo = &loc->halo;

   const int nxl = l1->nxl;
   const int nyl = l1->nyl;

   kernel = l1->krUpdate;
   retinaKernel = retina->krUpdate;

   //printf("Executing on device...");
   //kernel->run(nx*ny, nxl*nyl);

   // Check results for accuracy
   //
   //printf("\nChecking results...");
   //   d_rnd_state->copyFromDevice();
   //   status = check_results(rnd_state, rnd_state2, NX*NY);
   if (status != CL_SUCCESS) {
      exit(status);
   }

   //printf(" CORRECT\n");

   for (int n = 0; n < nWarm; n++) {
      kernel->run(nx*ny, nxl*nyl);
      retinaKernel->run(nx*ny, nxl*nyl);

   }

   printf("Timing %d loops on device... ", nLoops);
   timer.start();
   for (int n = 0; n < nLoops; n++) {
      kernel->run(nx*ny, nxl*nyl);
      retinaKernel->run(nx*ny, nxl*nyl);

   }
   status |= kernel->finish();
   timer.stop();
   timer.elapsed_time();

   l1->clActivity->copyFromDevice();
   // TODO - give kernel real input
   //for (int k = 0; k < 10; k++) {
   //   printf("A==%f\n", l1->clayer->activity->data[k]);
   //}

   printf("Timing %d loops on host..... ", nLoops);

   float time = l1->getParent()->simulationTime();
   float dt   = l1->getParent()->getDeltaTime();

   pvdata_t * gSynHead = l1->getChannel(CHANNEL_EXC); /*All channels allocated at once, so this array points to all channels*/
   pvdata_t * activity = l1->clayer->activity->data;

   timer.reset();
   timer.start();
   for (int n = 0; n < nLoops; n++) {
      LIF_update_state_arma(l1->getNumNeurons(), time, dt, nx, ny, nf,
            halo->lt, halo->rt, halo->dn, halo->up,
            &(l1->lParams), l1->rand_state,
            l1->getV(), l1->getVth(),
            l1->getConductance(CHANNEL_EXC),
            l1->getConductance(CHANNEL_INH),
            l1->getConductance(CHANNEL_INHB),
            gSynHead, activity);
   }
   timer.stop();
   timer.elapsed_time();

   // Shutdown and cleanup
   //
   //   clReleaseMemObject(d_rnd_state);
   //   delete cld;

   printf("Finished...\n");

   return status;
}

};

int main(int argc, char * argv[])
{
   return PV::test_kernels(argc, argv);
}

#ifdef NOTYET
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
   }
   return 0;
}

void c_rand(uint4 * rnd_state, int count)
{
   for (int k = 0; k < count; k++) {
      rnd_state[k] = cl_random_get(rnd_state[k]);
   }
}
#endif
