/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>

int checkForNonzero(HyPerCol * hc, int argc, char ** argv);

int main(int argc, char * argv[]) {

   int status;
   status = buildandrun(argc, argv, NULL, checkForNonzero, NULL, 0);
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int checkForNonzero(HyPerCol * hc, int argc, char ** argv) {
   // A customexit hook to verify that the comparison layer is zero
   // because of a nonzero on GsynExc cancelling the same value on
   // GSynInh, and not because some error caused Comparison to
   // return zero even if Output and CorrectValues were different.
   // A RequireAllZeroActivityProbe checks whether the activity is
   // zero everywhere, so this function doesn't have to.

   BaseLayer * basecomparisonlayer = hc->getLayerFromName("Comparison");
   HyPerLayer * comparisonlayer = dynamic_cast<HyPerLayer *>(basecomparisonlayer);
   assert(comparisonlayer!=NULL);
   assert(comparisonlayer->getNumChannels()==2);

   pvdata_t * exc = comparisonlayer->getChannel(CHANNEL_EXC);
   pvdata_t * inh = comparisonlayer->getChannel(CHANNEL_INH);
   int N = comparisonlayer->getNumNeurons();
   for (int n=0; n<N; n++) {
      assert(exc[n]!=0 && exc[n]==inh[n]);
   }
   return PV_SUCCESS;
}
