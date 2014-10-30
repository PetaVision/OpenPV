/*
 * ConnStatsProbe.cpp
 *
 *  Created on: Oct 27, 2014
 *      Author: pschultz
 */

#include "ConnStatsProbe.hpp"

namespace PV {

ConnStatsProbe::ConnStatsProbe(const char * probeName, HyPerCol * hc) {
   initialize_base();
   initialize(probeName, hc);
}

ConnStatsProbe::ConnStatsProbe() {
   initialize_base();
}

int ConnStatsProbe::initialize_base() {
   sums = NULL;
   sumabs = NULL;
   sumsquares = NULL;
   maxes = NULL;
   mins = NULL;
   maxabs = NULL;
   return PV_SUCCESS;
}

int ConnStatsProbe::initialize(const char * probeName, HyPerCol * hc) {
   int status = BaseConnectionProbe::initialize(probeName, hc);
   return status;
}

int ConnStatsProbe::allocateDataStructures() {
   HyPerConn * conn = this->getTargetHyPerConn();
   int numArbors = conn->numberOfAxonalArborLists();
   int numPatches = conn->getNumDataPatches();
   assert(numArbors>0 && numPatches>0);
   int numStats = (numArbors+(numArbors>1?1:0))*numPatches; // The (numArbors>1) on arbors is for the aggregate
   assert(numStats>0);
   int buffersize = numStats * (int) (3*(sizeof(double))+3*(sizeof(float))); // sums, sumabs, sumsquares are double; maxes, mins, maxabs are float
   statsptr = (char *) calloc((size_t) buffersize, sizeof(char));
   if (statsptr == NULL) {
      fprintf(stderr, "ConnStatsProbe \"%s\" error allocating buffer of size %d: %s\n",
            name, buffersize, strerror(errno));
      exit(EXIT_FAILURE);
   }
   sums = (double *) (statsptr);
   sumabs = (double *) (statsptr+sizeof(double)*numStats);
   sumsquares = (double *) (statsptr+2*sizeof(double)*numStats);
   maxes = (float *) (statsptr+3*sizeof(double)*numStats);
   mins = (float *) (statsptr+(3*sizeof(double)+sizeof(float))*numStats);
   maxabs = (float *) (statsptr+(3*sizeof(double)+2*sizeof(float))*numStats);
   return PV_SUCCESS;
}

int ConnStatsProbe::outputState(double simtime) {
   HyPerConn * conn = this->getTargetHyPerConn();
   int numArbors = conn->numberOfAxonalArborLists();
   int numPatches = conn->getNumDataPatches();
   int nxp = conn->xPatchSize();
   int nyp = conn->yPatchSize();
   int nfp = conn->fPatchSize();
   int patchsize = nxp*nyp*nfp;
   size_t buffersize = numPatches*(numArbors>1?1:0)*(3*sizeof(double)+3*sizeof(float));
   for (int kpatch=0; kpatch<numPatches; kpatch++) {
      int patchindex = kpatch*(numArbors+(numArbors>1?1:0));
      for (int arbor=0; arbor<numArbors; arbor++) {
         sums[patchindex+arbor] = 0.0;
         sumabs[patchindex+arbor] = 0.0;
         sumsquares[patchindex+arbor] = 0.0;
         maxes[patchindex+arbor] = -FLT_MAX;
         mins[patchindex+arbor] = FLT_MAX;
         maxabs[patchindex+arbor] = 0.0f;
         pvwdata_t * w = conn->get_wDataHead(arbor, patchindex);
         for (int p=0; p<patchsize; p++) {
            pvwdata_t wgt = w[p];
            sums[patchindex] += wgt;
            sumabs[patchindex] += fabs(wgt);
            sumsquares[patchindex] += wgt*wgt;
            if (maxes[patchindex] < wgt) { maxes[patchindex] = wgt; }
            if (mins[patchindex] > wgt) { mins[patchindex] = wgt; }
            if (maxabs[patchindex] < fabs(wgt)) { maxabs[patchindex] = fabs(wgt); }
         }
      }
      if (numArbors>1) {
         sums[patchindex+numArbors] = 0.0;
         sumabs[patchindex+numArbors] = 0.0;
         sumsquares[patchindex+numArbors] = 0.0;
         maxes[patchindex+numArbors] = -FLT_MAX;
         mins[patchindex+numArbors] = FLT_MAX;
         maxabs[patchindex+numArbors] = 0.0f;
         for (int arbor=0; arbor<numArbors; arbor++) {
            sums[patchindex+numArbors] += sums[patchindex+arbor];
            sumabs[patchindex+numArbors] += sumabs[patchindex+arbor];
            sumsquares[patchindex+numArbors] += sumsquares[patchindex+arbor];
            if (maxes[patchindex+numArbors] < maxes[patchindex+arbor]) { maxes[patchindex+numArbors]=maxes[patchindex+arbor]; }
            if (mins[patchindex+numArbors] > mins[patchindex+arbor]) { mins[patchindex+numArbors]=maxes[patchindex+arbor]; }
            if (maxabs[patchindex+numArbors] < maxabs[patchindex+arbor]) { maxabs[patchindex+numArbors]=maxabs[patchindex+arbor]; }
         }
      }

      InterColComm * icComm = parent->icCommunicator();
      if (parent->columnId()==0) {
         for (int proc=0; proc<icComm->commSize(); proc++) {
            if (proc!=0) {
               MPI_Recv(statsptr, buffersize, MPI_CHAR, proc, 203, icComm->communicator(), MPI_STATUS_IGNORE);
            }
            fprintf(outputstream->fp, "%s Process 0: \n", this->getMessage());
            for (int kpatch=0; kpatch<numPatches; kpatch++) {
               int patchindex = kpatch*(numArbors+(numArbors>1?1:0));
               for (int arbor=0; arbor<numArbors; arbor++) {
                  int idx=patchindex+arbor;
                  fprintf(outputstream->fp, "    Patch index %d, arbor %3d, sum=%f, L1-norm=%f, L2-norm=%f, max=%f, min=%f, max(abs)=%f\n",
                        kpatch, arbor, sums[idx], sumabs[idx], sqrt(sumsquares[idx]), maxes[idx], mins[idx], maxabs[idx]);
               }
               if (numArbors>1) {
                  int idx = patchindex+numArbors;
                  fprintf(outputstream->fp, "    Patch index %d, all arbors, sum=%f, L1-norm=%f, L2-norm=%f, max=%f, min=%f, max(abs)=%f\n",
                        kpatch, sums[idx], sumabs[idx], sqrt(sumsquares[idx]), maxes[idx], mins[idx], maxabs[idx]);
               }
            }
         }
      }
      else {
         MPI_Send(statsptr, buffersize, MPI_CHAR, 0, 203, icComm->communicator());
      }
   }
   return PV_SUCCESS;
}

ConnStatsProbe::~ConnStatsProbe() {
}

} /* namespace PV */
