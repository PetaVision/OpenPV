#include "ResetStateOnTriggerTestProbe.hpp"
#include <layers/HyPerLayer.hpp>

ResetStateOnTriggerTestProbe::ResetStateOnTriggerTestProbe(char const * probeName, PV::HyPerCol * hc) {
   initialize_base();
   initialize(probeName, hc);
}

ResetStateOnTriggerTestProbe::ResetStateOnTriggerTestProbe() {
   initialize_base();
}

int ResetStateOnTriggerTestProbe::initialize_base() {
   probeStatus = 0;
   firstFailureTime = 0;
   return PV_SUCCESS;
}

int ResetStateOnTriggerTestProbe::initialize(char const * probeName, PV::HyPerCol * hc) {
   int status = PV_SUCCESS;
   status = PV::LayerProbe::initialize(probeName, hc);
   return status;
}

int ResetStateOnTriggerTestProbe::calcValues(double timevalue) {
   int nBatch = getNumValues();
   if (timevalue > parent->getStartTime()) {
      int N = targetLayer->getNumNeurons();
      int NGlobal = targetLayer->getNumGlobalNeurons();
      PVLayerLoc const * loc = targetLayer->getLayerLoc();
      PVHalo const * halo = &loc->halo;
      int inttime = (int) nearbyintf(timevalue/parent->getDeltaTime());
      for (int b=0; b<nBatch; b++) {
         int numDiscreps = 0;
         pvadata_t const * activity = targetLayer->getLayerData() + b*targetLayer->getNumExtended();
         for (int k=0; k<N; k++) {
            int kex = kIndexExtended(k, loc->nx, loc->ny, loc->nf, halo->lt, halo->rt, halo->dn, halo->up);
            pvadata_t a = activity[kex];
            int kGlobal = globalIndexFromLocal(k, *loc);
            int correctValue = 4*kGlobal*((inttime + 4)%5+1) + (kGlobal==((((inttime-1)/5)*5)+1)%NGlobal);
            if ( a != (pvadata_t) correctValue ) { numDiscreps++; }
         }
         getValuesBuffer()[b] = (double) numDiscreps;
      }
      MPI_Allreduce(MPI_IN_PLACE, getValuesBuffer(), nBatch, MPI_DOUBLE, MPI_SUM, parent->icCommunicator()->communicator());
      if (probeStatus==0) {
         for (int k=0; k<nBatch; k++) {
            if (getValuesBuffer()[k]) {
               probeStatus = 1;
               firstFailureTime = timevalue;
            }
         }
      }      
   }
   else {
      for (int b=0; b<nBatch; b++) {
         getValuesBuffer()[b] = 0.0;
      }
   }
   return PV_SUCCESS;
}

int ResetStateOnTriggerTestProbe::outputState(double timevalue) {
   getValues(timevalue); // calls calcValues
   if (parent->columnId()!=0) { return PV_SUCCESS; }
   if (probeStatus != 0) {
      int nBatch = getNumValues();
      if (nBatch==1) {
         int nnz = (int) nearbyint(getValuesBuffer()[0]);
         fprintf(outputstream->fp, "%s t=%f, %d neuron%s the wrong value.\n",
               getMessage(), timevalue, nnz, nnz==1 ? " has" : "s have");
      }
      else {
         for (int k=0; k<nBatch; k++) {
            int nnz = (int) nearbyint(getValuesBuffer()[k]);
            fprintf(outputstream->fp, "%s t=%f, batch element %d, %d neuron%s the wrong value.\n",
                  getMessage(), timevalue, k, nnz, nnz==1 ? " has" : "s have");
         }
      }
   }
   return PV_SUCCESS;
}

ResetStateOnTriggerTestProbe::~ResetStateOnTriggerTestProbe() {
}

PV::BaseObject * createResetStateOnTriggerTestProbe(char const * name, PV::HyPerCol * hc) { 
   return hc ? new ResetStateOnTriggerTestProbe(name, hc) : NULL;
}

