/*
 * pv.cpp
 *
 */


#include "../PetaVision/src/columns/buildandrun.hpp"
#include "KernelTestProbe.hpp"

#define MAIN_USES_ADDCUSTOM

#ifdef MAIN_USES_ADDCUSTOM
int addcustom(HyPerCol * hc, int argc, char * argv[]);
// addcustom is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

void * customgroup(const char * keyword, const char * name, HyPerCol * hc);

int main(int argc, char * argv[]) {

   int status;
#ifdef MAIN_USES_ADDCUSTOM
   status = buildandrun(argc, argv, &addcustom);
#else
   status = buildandrun(argc, argv);
#endif // MAIN_USES_ADDCUSTOM
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_ADDCUSTOM
int addcustom(HyPerCol * hc, int argc, char * argv[]) {
   int status;
   PVParams * params = hc->parameters();
   int numGroups = params->numberOfGroups();
   for (int n = 0; n < numGroups; n++) {
      const char * kw = params->groupKeywordFromIndex(n);
      const char * name = params->groupNameFromIndex(n);
      HyPerLayer * targetlayer;
      const char * message;
      const char * filename;
      KernelTestProbe * addedProbe;
      if (!strcmp(kw, "KernelTestProbe")) {
         status = getLayerFunctionProbeParameters(name, kw, hc, &targetlayer,
               &message, &filename);
         if (status != PV_SUCCESS) {
            fprintf(stderr, "Skipping params group \"%s\"\n", name);
            continue;
         }
         if( filename ) {
            addedProbe =  new KernelTestProbe(filename, hc, message);
         }
         else {
            addedProbe =  new KernelTestProbe(message);
         }
         if( !addedProbe ) {
            fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
         }
         assert(targetlayer);
         if( addedProbe ) targetlayer->insertProbe(addedProbe);
         checknewobject((void *) addedProbe, kw, name, hc);
      }
   }
   return PV_SUCCESS;
}
#endif // MAIN_USES_ADDCUSTOM

void * customgroup(const char * keyword, const char * name, HyPerCol * hc) {
   int status;
   // PVParams * params = hc->parameters();
   // HyPerLayer * preLayer;
   // HyPerLayer * postLayer;
   LayerProbe * addedProbe;
   void * addedGroup = NULL;
   const char * filename;
   HyPerLayer * targetlayer;
   const char * message;
   bool errorFound;
   if( !strcmp(keyword, "KernelTestProbe") ) {
      status = getLayerFunctionProbeParameters(name, keyword, hc, &targetlayer, &message, &filename);
      errorFound = status!=PV_SUCCESS;
      if( !errorFound ) {
         PVBufType buf_type = BufV;
         if (targetlayer->getSpikingFlag()) {
            buf_type = BufActivity;
         }
         if( filename ) {
            addedProbe = (LayerProbe *) new KernelTestProbe(filename, hc, message);
         }
         else {
            addedProbe = (LayerProbe *) new KernelTestProbe(message);
         }
         if( !addedProbe ) {
            fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
            errorFound = true;
         }
         if( !errorFound ) addedGroup = (void *) addedProbe;
      }
   }
   return addedGroup;
}

