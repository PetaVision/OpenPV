#include "MomentumTestConnProbe.hpp"

namespace PV {

MomentumTestConnProbe::MomentumTestConnProbe(const char * probename, HyPerCol * hc) {
   initialize_base();
   int status = initialize(probename, hc);
   pvErrorIf(!(status == PV_SUCCESS), "Test failed.\n");
}

MomentumTestConnProbe::MomentumTestConnProbe() {
   initialize_base();
}

MomentumTestConnProbe::~MomentumTestConnProbe() {
}

int MomentumTestConnProbe::initialize_base() {
   return PV_SUCCESS;
}

int MomentumTestConnProbe::initNumValues() {
   return setNumValues(-1);
}

int MomentumTestConnProbe::outputState(double timed){
   //Grab weights of probe and test for the value of .625/1.5, or .4166666
   HyPerConn* conn = getTargetHyPerConn();
   int numPreExt = conn->preSynapticLayer()->getNumExtended();
   int syw = conn->yPatchStride();                   // stride in patch

   for(int kPre = 0; kPre < numPreExt; kPre++){
      PVPatch * weights = conn->getWeights(kPre, 0);
      int nk  = conn->fPatchSize() * weights->nx;

      pvwdata_t * data = conn->get_wData(0,kPre);
      int ny  = weights->ny;
      pvdata_t wCorrect;
      for (int y = 0; y < ny; y++) {
         pvwdata_t * dataYStart = data + y * syw;
         for(int k = 0; k < nk; k++){
            pvdata_t wObserved = dataYStart[k];
            if(timed < 3){
               wCorrect = 0;
            }
            else{
               wCorrect = 0.376471f;
               for(int i = 0; i < (timed-3); i++){
                  wCorrect += 0.376471f * expf(-(2*(i+1)));
               }
            }
            pvErrorIf(!(fabsf(wObserved - wCorrect) <= 1e-4f), "Test failed.\n");
         }
      }
   }
   return PV_SUCCESS;

}

}  // end of namespace PV


