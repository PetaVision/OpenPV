#include "MomentumTestConnProbe.hpp"

namespace PV {

MomentumTestConnProbe::MomentumTestConnProbe(const char * probename, HyPerCol * hc) {
   initialize_base();
   int status = initialize(probename, hc);
   assert(status == PV_SUCCESS);
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
               wCorrect = .376471;
               for(int i = 0; i < (timed-3); i++){
                  wCorrect += .376471 * exp(-(2*(i+1)));
               }
            }
            assert(fabs(wObserved - wCorrect) <= 1e-4);
         }
      }
   }
   return PV_SUCCESS;

}

BaseObject * createMomentumTestConnProbe(char const * name, HyPerCol * hc) {
   return hc ? new MomentumTestConnProbe(name, hc) : NULL;
}

}  // end of namespace PV


