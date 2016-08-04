#include "TestConnProbe.hpp"

namespace PV {

TestConnProbe::TestConnProbe(const char * probename, HyPerCol * hc) {
   initialize_base();
   int status = initialize(probename, hc);
   pvErrorIf(!(status == PV_SUCCESS), "Test failed.\n");
}

TestConnProbe::TestConnProbe() {
   initialize_base();
}

TestConnProbe::~TestConnProbe() {
}

int TestConnProbe::initialize_base() {
   return PV_SUCCESS;
}

int TestConnProbe::initNumValues() {
   return setNumValues(-1);
}

int TestConnProbe::outputState(double timed){
   //Grab weights of probe and test for the value of .625/1.5, or .4166666
   HyPerConn* conn = getTargetHyPerConn();
   int numPreExt = conn->preSynapticLayer()->getNumExtended();
   int syw = conn->yPatchStride();                   // stride in patch

   for(int kPre = 0; kPre < numPreExt; kPre++){
      PVPatch * weights = conn->getWeights(kPre, 0);
      int nk  = conn->fPatchSize() * weights->nx;

      pvwdata_t * data = conn->get_wData(0,kPre);
      int ny  = weights->ny;
      for (int y = 0; y < ny; y++) {
         pvwdata_t * dataYStart = data + y * syw;
         for(int k = 0; k < nk; k++){
            if(fabs(timed - 0) < (parent->getDeltaTime()/2)){
               if(fabs(dataYStart[k] - 1) > .01){
                  pvError() << "dataYStart[k]: " << dataYStart[k] << "\n";
               }
               pvErrorIf(!(fabs(dataYStart[k] - 1) <= .01), "Test failed.\n");
            }
            else if(fabs(timed - 1) < (parent->getDeltaTime()/2)){
               if(fabs(dataYStart[k] - 1.375) > .01){
                  pvError() << "dataYStart[k]: " << dataYStart[k] << "\n";
               }
               pvErrorIf(!(fabs(dataYStart[k] - 1.375) <= .01), "Test failed.\n");
            }

         }
      }
   }
   return PV_SUCCESS;

}

}  // end of namespace PV


