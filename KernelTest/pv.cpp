/*
 * pv.cpp
 *
 */


#include "../PetaVision/src/columns/buildandrun.hpp"
#include "KernelTestProbe.hpp"

#undef MAIN_USES_ADDCUSTOM

#ifdef MAIN_USES_ADDCUSTOM
int addcustom(HyPerCol * hc, int argc, char * argv[]);
// addcustom is for adding objects not supported by build().
#endif // MAIN_USES_ADDCUSTOM

void * customgroup(const char * keyword, const char * name, HyPerCol * hc);

int main(int argc, char * argv[]) {

   int status;
#ifdef MAIN_USES_ADDCUSTOM
   status = buildandrun(argc, argv, &addcustom, NULL, &customgroup);
#else
   status = buildandrun(argc, argv, NULL, NULL, &customgroup);
#endif // MAIN_USES_ADDCUSTOM
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef MAIN_USES_ADDCUSTOM
int addcustom(HyPerCol * hc, int argc, char * argv[]) {
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
   char * message = NULL;
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
            addedProbe = (LayerProbe *) new KernelTestProbe(filename, targetlayer, message);
         }
         else {
            addedProbe = (LayerProbe *) new KernelTestProbe(targetlayer, message);
         }
         free(message); message=NULL; // message was alloc'ed in getLayerFunctionProbeParameters call
         if( !addedProbe ) {
            fprintf(stderr, "Group \"%s\": Unable to create probe\n", name);
            errorFound = true;
         }
         if( !errorFound ) addedGroup = (void *) addedProbe;
      }
   }
   return addedGroup;
}

