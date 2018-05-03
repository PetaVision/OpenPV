/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <layers/HyPerLayer.hpp>

int customexit(HyPerCol *hc, int argc, char **argv);

int main(int argc, char *argv[]) {
   int status;
   status = buildandrun(argc, argv, NULL /*custominit*/, &customexit);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol *hc, int argc, char **argv) {
   float correctvalue = 0.5f;
   float tolerance    = 1.0e-3f;

   if (hc->columnId() == 0) {
      InfoLog().printf(
            "Checking whether input layer has all values equal to %f ...\n", (double)correctvalue);
   }
   HyPerLayer *inputlayer = dynamic_cast<HyPerLayer *>(hc->getObjectFromName("input"));
   FatalIf(!(inputlayer), "Test failed.\n");
   PVLayerLoc const *loc = inputlayer->getLayerLoc();
   FatalIf(!(loc->nf == 1), "Test failed.\n");
   const int numNeurons = inputlayer->getNumNeurons();
   FatalIf(!(numNeurons > 0), "Test failed.\n");
   int status = PV_SUCCESS;

   int numExtended        = inputlayer->getNumExtended();
   Communicator *icComm   = hc->getCommunicator();
   float const *layerData = inputlayer->getLayerData();
   int rootproc           = 0;
   if (icComm->commRank() == rootproc) {
      float *databuffer = (float *)malloc(numExtended * sizeof(float));
      FatalIf(!(databuffer), "Test failed.\n");
      for (int proc = 0; proc < icComm->commSize(); proc++) {
         if (proc == rootproc) {
            memcpy(databuffer, layerData, numExtended * sizeof(float));
         }
         else {
            MPI_Recv(
                  databuffer,
                  numExtended * sizeof(float),
                  MPI_BYTE,
                  proc,
                  15,
                  icComm->communicator(),
                  MPI_STATUS_IGNORE);
         }
         // At this point, databuffer on rank 0 should contain the extended input layer on rank proc
         for (int k = 0; k < numNeurons; k++) {
            int kExt = kIndexExtended(
                  k,
                  loc->nx,
                  loc->ny,
                  loc->nf,
                  loc->halo.lt,
                  loc->halo.rt,
                  loc->halo.dn,
                  loc->halo.up);
            float value = databuffer[kExt];
            if (fabsf(value - correctvalue) >= tolerance) {
               ErrorLog().printf(
                     "Rank %d, restricted index %d, extended index %d, value is %f instead of %f\n",
                     proc,
                     k,
                     kExt,
                     (double)value,
                     (double)correctvalue);
               status = PV_FAILURE;
            }
         }
      }
      free(databuffer);
      if (status == PV_SUCCESS) {
         InfoLog().printf("%s succeeded.\n", argv[0]);
      }
      else {
         Fatal().printf("%s failed.\n", argv[0]);
      }
   }
   else {
      // const_cast necessary because older versions of MPI define MPI_Send with first arg as void*,
      // not void const*.
      MPI_Send(
            const_cast<float *>(layerData),
            numExtended * sizeof(float),
            MPI_BYTE,
            rootproc,
            15,
            icComm->communicator());
   }
   MPI_Barrier(icComm->communicator());
   return status;
}
