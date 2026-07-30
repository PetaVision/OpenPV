/*
 * pv.cpp
 *
 */

#include <columns/buildandrun.hpp>
#include <components/BasePublisherComponent.hpp>
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
   FatalIf(inputlayer == nullptr, "No layer named \"input\".\n");
   auto *inputpublisher  = inputlayer->getComponentByType<BasePublisherComponent>();
   PVLayerLoc const *loc = inputpublisher->getLayerLoc();
   FatalIf(loc->nf != 1, "Layer \"input\" nf must be 1 (values is %d).\n", loc->nf);
   const long numNeurons = (long)loc->nx * (long)loc->ny * (long)loc->nf;
   FatalIf(numNeurons <= 0, "Test failed.\n");
   int status = PV_SUCCESS;

   std::size_t numExtended = static_cast<std::size_t>(inputpublisher->getNumExtended());
   std::size_t numBytesL   = numExtended * sizeof(float);
   int numBytes            = static_cast<int>(numBytesL);
   FatalIf(
         static_cast<std::size_t>(numBytes) != numBytesL,
         "Buffer is %ld bytes, which is too big for MPI send/receive.\n",
         numBytesL);
   Communicator *icComm   = hc->getCommunicator();
   float const *layerData = inputpublisher->getLayerData();
   int rootproc           = 0;
   if (icComm->commRank() == rootproc) {
      float *databuffer = (float *)malloc(numBytesL);
      FatalIf(!(databuffer), "Test failed.\n");
      for (int proc = 0; proc < icComm->commSize(); proc++) {
         if (proc == rootproc) {
            memcpy(databuffer, layerData, numBytesL);
         }
         else {
            MPI_Recv(
                  databuffer,
                  numBytes,
                  MPI_BYTE,
                  proc,
                  15,
                  icComm->communicator(),
                  MPI_STATUS_IGNORE);
         }
         // At this point, databuffer on rank 0 should contain the extended input layer on rank proc
         for (long k = 0; k < numNeurons; k++) {
            long kExt = kIndexExtended(
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
                     "Rank %d, restricted index %ld, extended index %ld, value is %f instead of %f\n",
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
            numBytes,
            MPI_BYTE,
            rootproc,
            15,
            icComm->communicator());
   }
   MPI_Barrier(icComm->communicator());
   return status;
}
