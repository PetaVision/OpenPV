/*
 * PV_Init.cpp
 *
 *  Created on: Jul 31, 2015
 *      Author: slundquist
 */
#include <csignal>
#include <cMakeHeader.h>
#ifdef PV_USE_OPENMP_THREADS
#include <omp.h>
#endif // PV_USE_OPENMP_THREADS
#include "PV_Init.hpp"
#include "columns/HyPerCol.hpp"

namespace PV {

PV_Init::PV_Init(int* argc, char ** argv[], bool allowUnrecognizedArguments){
   //Initialize MPI
   initSignalHandler();
   commInit(argc, argv);
   initMaxThreads();
   params = NULL;
   icComm = NULL;
   arguments = new PV_Arguments(*argc, *argv, allowUnrecognizedArguments);
   factory = new Factory();
   initialized = false;
   buildandrunDeprecationWarning = true;
}

PV_Init::~PV_Init(){
   delete params;
   delete icComm;
   delete arguments;
   delete factory;
   commFinalize();
}

int PV_Init::initSignalHandler()
{
   // Block SIGUSR1.  root process checks for SIGUSR1 during advanceTime() and broadcasts sends to all processes,
   // which saves the result in the checkpointSignal member variable.
   // When run() checks whether to call checkpointWrite, it looks at checkpointSignal, and writes a
   // checkpoint if checkpointWriteFlag is true, regardless of whether the next scheduled checkpoint time has arrived.
   //
   // This routine must be called before MPI_Initialize; otherwise a thread created by MPI will not get the signal handler
   // but will get the signal and the job will terminate.
   sigset_t blockusr1;
   sigemptyset(&blockusr1);
   sigaddset(&blockusr1, SIGUSR1);
   sigprocmask(SIG_BLOCK, &blockusr1, NULL);
   return 0;
}

int PV_Init::initialize() {
   delete icComm;
   icComm = new InterColComm(arguments);
   initialized = true;
   int status = PV_SUCCESS;
   // It is okay to initialize without there being a params file.
   // setParams() can be called later.
   delete params; params = NULL;
   if (arguments->getParamsFile()) {
      status = createParams();
   }
   return status;
}

int PV_Init::initMaxThreads() {
#ifdef PV_USE_OPENMP_THREADS
   maxThreads = omp_get_max_threads();
#else // PV_USE_OPENMP_THREADS
   maxThreads = 1;
#endif // PV_USE_OPENMP_THREADS
   return PV_SUCCESS;
}

int PV_Init::commInit(int* argc, char*** argv)
{
   int mpiInit;
   // If MPI wasn't initialized, initialize it.
   // Remember if it was initialized on entry; the destructor will only finalize if the constructor init'ed.
   // This way, you can do several simulations sequentially by initializing MPI before creating
   // the first HyPerCol; after running the first simulation the MPI environment will still exist and you
   // can run the second simulation, etc.
   MPI_Initialized(&mpiInit);
   if( !mpiInit) {
      assert((*argv)[*argc]==NULL); // Open MPI 1.7 assumes this.
      MPI_Init(argc, argv);
   }
   else{
      std::cout << "Error: PV_Init communicator already initialized\n";
      exit(-1);
   }

   return 0;
}

int PV_Init::setParams(char const * params_file) {
   if (params_file == NULL) { return PV_FAILURE; }
   char const * newParamsFile = getArguments()->setParamsFile(params_file);
   if (newParamsFile==NULL) {
      fflush(stdout);
      fprintf(stderr, "PV_Init error setting new params file: %s\n", strerror(errno));
      return PV_FAILURE;
   }
   if (!initialized) { initialize(); }
   return createParams();
   return PV_SUCCESS;
}

int PV_Init::createParams() {
   assert(initialized);
   char const * params_file = getArguments()->getParamsFile();
   if (params_file) {
      delete params;
      params = new PVParams(params_file, 2*(INITIAL_LAYER_ARRAY_SIZE+INITIAL_CONNECTION_ARRAY_SIZE), icComm);
      return PV_SUCCESS;
   }
   else {
      return PV_FAILURE;
   }
}

int PV_Init::registerKeyword(char const * keyword, ObjectCreateFn creator) {
   int status = factory->registerKeyword(keyword, creator);
   if (status != PV_SUCCESS) {
      if (getWorldRank()==0) {
         fflush(stdout);
         fprintf(stderr, "PV_Init error: keyword \"%s\" has already been registered.\n", keyword);
      }
   }
   return status;
}

BaseObject * PV_Init::create(char const * keyword, char const * name, HyPerCol * hc) const {
   BaseObject * pvObject = factory->create(keyword, name, hc);
   if (pvObject == NULL) {
      if (getWorldRank()==0) {
         fflush(stdout);
         fprintf(stderr, "Unable to create %s \"%s\": keyword \"%s\" has not been registered.\n", keyword, name, keyword);
      }
   }
   return pvObject;
}

HyPerCol * PV_Init::build() {
   HyPerCol * hc = new HyPerCol("column", this);
   if( hc == NULL ) {
      fprintf(stderr, "Unable to create HyPerCol\n");
      return NULL;
   }
   PVParams * hcparams = hc->parameters();
   int numGroups = hcparams->numberOfGroups();

   // Make sure first group defines a column
   if( strcmp(hcparams->groupKeywordFromIndex(0), "HyPerCol") ) {
      fprintf(stderr, "First group in the params file did not define a HyPerCol.\n");
      delete hc;
      return NULL;
   }

   for (int k=0; k<numGroups; k++) {
      const char * kw = hcparams->groupKeywordFromIndex(k);
      const char * name = hcparams->groupNameFromIndex(k);
      if (!strcmp(kw, "HyPerCol")) {
         if (k==0) { continue; }
         else {
            if (hc->columnId()==0) {
               fflush(stdout);
               fprintf(stderr, "Group %d in params file (\"%s\") is a HyPerCol; the HyPerCol must be the first group.\n",
                       k+1, name);
            }
            delete hc;
            return NULL;
         }
      }
      else {
         BaseObject * addedObject = factory->create(kw, name, hc);
         if (addedObject==NULL) {
            if (hc->globalRank()==0) {
               fflush(stdout);
               fprintf(stderr, "Error creating %s \"%s\".\n", kw, name);
            }
            delete hc;
            return NULL;
         }
      }
   }

   if( hc->numberOfLayers() == 0 ) {
      fflush(stdout);
      fprintf(stderr, "HyPerCol \"%s\" does not have any layers.\n", hc->getName());
      delete hc;
      return NULL;
   }
   return hc;
}

int PV_Init::commFinalize()
{
   MPI_Finalize();
   return 0;
}

void PV_Init::printBuildAndRunDeprecationWarning(char const * functionName, char const * functionSignature) {
   if (buildandrunDeprecationWarning) {
      if (getWorldRank()==0) {
         fprintf(stderr, "\nWarning: %s(%s) has been deprecated.  Use the Factory version of %s instead.\n\n",
               functionName, functionSignature, functionName);
      }
      clearBuildAndRunDeprecationWarning();
   }
}

} // namespace PV



