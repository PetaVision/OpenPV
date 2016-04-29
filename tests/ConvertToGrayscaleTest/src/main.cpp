/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>

int customexit(HyPerCol * hc, int argc, char ** argv);

int main(int argc, char * argv[]) {
   int status;
   status = buildandrun(argc, argv, NULL/*custominit*/, &customexit);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int customexit(HyPerCol * hc, int argc, char ** argv) {
   pvadata_t correctvalue = 0.5f;
   pvadata_t tolerance = 1.0e-7f;

   if (hc->columnId()==0) {
      printf("Checking whether input layer has all values equal to %f ...\n", correctvalue);
   }
   HyPerLayer * inputlayer = hc->getLayerFromName("input");
   assert(inputlayer);
   PVLayerLoc const * loc = inputlayer->getLayerLoc();
   assert(loc->nf==1);
   const int numNeurons = inputlayer->getNumNeurons();
   assert(numNeurons>0);
   int status = PV_SUCCESS;

   int numExtended = inputlayer->getNumExtended();
   InterColComm * icComm = hc->icCommunicator();
   pvadata_t * layerData = (pvadata_t *) icComm->publisherStore(inputlayer->getLayerId())->buffer(LOCAL);
   int rootproc = 0;
   if (icComm->commRank()==rootproc) {
      pvadata_t * databuffer = (pvadata_t *) malloc(numExtended*sizeof(pvadata_t));
      assert(databuffer);
      for (int proc=0; proc<icComm->commSize(); proc++) {
         if (proc==rootproc) {
            memcpy(databuffer, layerData, numExtended*sizeof(pvadata_t));
         }
         else {
            MPI_Recv(databuffer, numExtended*sizeof(pvadata_t),MPI_BYTE,proc,15,icComm->communicator(), MPI_STATUS_IGNORE);
         }
         // At this point, databuffer on rank 0 should contain the extended input layer on rank proc
         for (int k=0; k<numNeurons; k++) {
            int kExt = kIndexExtended(k,loc->nx,loc->ny,loc->nf,loc->halo.lt,loc->halo.rt,loc->halo.dn,loc->halo.up);
            pvadata_t value = databuffer[kExt];
            if (fabs(value-correctvalue)>=tolerance) {
               fprintf(stderr, "Rank %d, restricted index %d, extended index %d, value is %f instead of %f\n",
                     proc, k, kExt, value, correctvalue);
               status = PV_FAILURE;
            }
         }
      }
      free(databuffer);
      if (status == PV_SUCCESS) {
         printf("%s succeeded.\n", argv[0]);
      }
      else {
         fprintf(stderr, "%s failed.\n", argv[0]);
         exit(status);
      }
   }
   else {
      MPI_Send(layerData,numExtended*sizeof(pvadata_t),MPI_BYTE,rootproc,15,icComm->communicator());
   }
   MPI_Barrier(icComm->communicator());
   return status;
}
