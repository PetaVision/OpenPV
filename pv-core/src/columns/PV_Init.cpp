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

namespace PV {

PV_Init::PV_Init(int* argc, char ** argv[], bool allowUnrecognizedArguments){
   //Initialize MPI
   initSignalHandler();
   commInit(argc, argv);
   initMaxThreads();
   params = NULL;
   icComm = NULL;
   arguments = new PV_Arguments(*argc, *argv, allowUnrecognizedArguments);
   initialized = false;
}

PV_Init::~PV_Init(){
   delete params;
   delete icComm;
   delete arguments;
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
   delete params; params=NULL;
   char const * params_file = arguments->getParamsFile();
   if (params_file) {
      setParams(params_file);
   }
   initialized = true;
   return PV_SUCCESS;
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
   PVParams * oldParams = params;
   params = new PVParams(params_file, 2*(INITIAL_LAYER_ARRAY_SIZE+INITIAL_CONNECTION_ARRAY_SIZE), icComm);
   delete oldParams;
   return PV_SUCCESS;
}

int PV_Init::commFinalize()
{
   MPI_Finalize();
   return 0;
}

} // namespace PV



