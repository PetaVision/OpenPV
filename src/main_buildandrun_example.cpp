/*
 * main_buildandrun_example.cpp
 *
 * An example of a main function that uses PetaVision's buildandrun and custom param group handlers.
 * Copy this file to some directory.  $PETAVISION should be the path to the directory that contains the src directory for PetaVision.
 *
 * Run the commands
 * mpicxx main_buildandrun_example.cpp -I$PETAVISION/src -c -o main_buildandrun_example.cpp.o
 * mpicxx -lgdal -lpv -L$PETAVISION/lib main_buildandrun_example.cpp.o -o main_build_andrun_example
 *
 * The above should compile without errors
 */

#include <columns/buildandrun.hpp>
#include <io/ParamGroupHandler.hpp>
#include <stdio.h>

#define BUILDANDRUN_USES_ADDL_ARGS

#ifdef BUILDANDRUN_USES_ADDL_ARGS
int custominit(PV::HyPerCol * hc, int argc, char * argv[]);
int customexit(PV::HyPerCol * hc, int argc, char * argv[]);

// An example of a custom layer derived from HyPerLayer
class CustomLayer : public PV::HyPerLayer {
public:
   CustomLayer(char const * name, PV::HyPerCol * hc) { initialize_base(); initialize(name, hc); }
   virtual ~CustomLayer() {}
   virtual bool activityIsSpiking() { return true; }

protected:
   CustomLayer() {}
   int initialize(char const * name, PV::HyPerCol * hc) { return PV::HyPerLayer::initialize(name, hc); }

protected:
   int initialize_base() { return PV_SUCCESS; }
};

// An example of a custom connection derived from HyPerConn
class CustomConn : public PV::HyPerConn {
public:
   CustomConn(char const * name, PV::HyPerCol * hc, PV::InitWeights * weightInitializer, PV::NormalizeBase * weightNormalizer) { initialize_base(); initialize(name, hc, weightInitializer, weightNormalizer); }
   virtual ~CustomConn() {}

protected:
   CustomConn() {}
   int initialize(char const * name, PV::HyPerCol * hc, PV::InitWeights * weightInitializer, PV::NormalizeBase * weightNormalizer) { return PV::HyPerConn::initialize(name, hc, weightInitializer, weightNormalizer); }

protected:
   int initialize_base() { return PV_SUCCESS; }
};

// An example of a custom group handler that recognizes a custom layer.
class groupHandler1 : public PV::ParamGroupHandler {
public:
   groupHandler1() {}
   virtual ~groupHandler1() {}
   virtual PV::ParamGroupType getGroupType(char const * keyword) {
      if (keyword == NULL) { return UnrecognizedGroupType; }
      else if (!strcmp(keyword, "CustomLayer")) { return PV::LayerGroupType; }
      else { return UnrecognizedGroupType; }
   }
   virtual HyPerLayer * createLayer(char const * keyword, char const * name, PV::HyPerCol * hc) {
      if (keyword == NULL) { return NULL; }
      else if (!strcmp(keyword, "CustomLayer")) { return new CustomLayer(name, hc); }
      else { return NULL; }
   }
};

// An example of a custom group handler that recognizes a custom connection.
class groupHandler2 : public PV::ParamGroupHandler {
public:
   groupHandler2() {}
   virtual ~groupHandler2() {}
   virtual ParamGroupType getGroupType(char const * keyword) {
      if (keyword == NULL) { return UnrecognizedGroupType; }
      else if (!strcmp(keyword, "CustomConn")) { return ConnectionGroupType; }
      else { return UnrecognizedGroupType; }
   }
   virtual BaseConnection * createConnection(char const * keyword, char const * name, PV::HyPerCol * hc, PV::InitWeights * weightInitializer, PV::NormalizeBase * weightNormalizer) {
      if (keyword == NULL) { return NULL; }
      else if (!strcmp(keyword, "CustomConn")) { return new CustomConn(name, hc, weightInitializer, weightNormalizer); }
      else { return NULL; }
   }
};

#endif // MAIN_USES_ADDCUSTOM

int main(int argc, char * argv[]) {

   int status;
#ifdef BUILDANDRUN_USES_ADDL_ARGS
   int const numGroupHandlers = 2;
   PV::ParamGroupHandler * customGroupHandlers[numGroupHandlers];
   customGroupHandlers[0] = new groupHandler1;
   customGroupHandlers[1] = new groupHandler2;

   status = buildandrun(argc, argv, &custominit, &customexit, (PV::ParamGroupHandler **) customGroupHandlers, numGroupHandlers);
#else
   status = buildandrun(argc, argv);
#endif // BUILDANDRUN_USES_ADDL_ARGS
   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#ifdef BUILDANDRUN_USES_ADDL_ARGS
int custominit(PV::HyPerCol * hc, int argc, char * argv[]) {
   printf("custominit(hc, argc, argv) is called after building the HyPerCol\n");
   printf("and before calling HyPerCol::run to run the simulation.\n");
   printf("Hence custominit can be used to do any specialized initialization tasks.\n");

   return PV_SUCCESS;
}

int customexit(PV::HyPerCol * hc, int argc, char * argv[]) {
   printf("customexit(hc, argc, argv) is called after calling HyPerCol::run\n");
   printf("and before deleting the HyPerCol.\n");
   printf("Hence customexit can be used to do any specialized finalization tasks.\n");

   return PV_SUCCESS;
}

#endif // MAIN_USES_ADDCUSTOM
