/*
 * PV_Init.cpp
 *
 *  Created on: Jul 31, 2015
 *      Author: slundquist
 */
#include "PV_Init.hpp"
#include "cMakeHeader.h"
#include "columns/CoreKeywords.hpp"
#include "columns/Factory.hpp"
#include "include/pv_common.h"
#include "io/io.hpp"
#include "structures/MPIBlock.hpp"
#include "utils/PVAlloc.hpp"
#include "utils/PVAssert.hpp"
#include "utils/PVLog.hpp"
#include "utils/PathComponents.hpp"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ios>
#include <sys/utsname.h>
#include <unistd.h>
#ifdef PV_USE_OPENMP_THREADS
#include <omp.h>
#endif // PV_USE_OPENMP_THREADS

namespace PV {

PV_Init::PV_Init(int *argc, char **argv[], bool allowUnrecognizedArguments) {
   // Initialize MPI
   initSignalHandler();
   commInit(argc, argv);
   initMaxThreads();
   mArgC = *argc;
   mArgV.resize(mArgC + 1);
   for (int a = 0; a < mArgC; a++) {
      mArgV[a] = argv[0][a];
   }
   params        = nullptr;
   mCommunicator = nullptr;

   mArguments = parse_arguments(*argc, *argv, allowUnrecognizedArguments);
   initLogFile(false /*appendFlag*/);
   initFactory();
   initialize(); // must be called after initialization of Arguments data member.
}

PV_Init::~PV_Init() {
   delete params;
   delete mCommunicator;
   mArguments = nullptr;
   commFinalize();
}

int PV_Init::initSignalHandler() {
   // Block SIGUSR1, SIGUSR2.  root process checks for
   // these signals during Checkpointer::checkpointWrite() (typically called
   // during HyPerCol::advanceTime()) and broadcasts any caught signal to
   // all processes.
   // CheckpointWrite() responds to the signals as follows:
   // SIGUSR1: write a checkpoint and continue.
   // SIGUSR2: write a checkpoint and quit.
   //
   // This routine must be called before MPI_Initialize; otherwise a thread
   // created by MPI will not get the signal handler
   // but will get the signal and the job will terminate.
   sigset_t blockusr1;
   sigemptyset(&blockusr1);
   sigaddset(&blockusr1, SIGUSR1);
   sigaddset(&blockusr1, SIGUSR2);
   sigprocmask(SIG_BLOCK, &blockusr1, nullptr);
   return 0;
}

int PV_Init::initialize() {
   delete mCommunicator;
   mCommunicator = new Communicator(mArguments.get());
   printInitMessage();
   int status    = PV_SUCCESS;
   // It is okay to initialize without there being a params file.
   // setParams() can be called later.
   delete params;
   params                 = nullptr;
   std::string paramsFile = mArguments->getStringArgument("ParamsFile");
   if (!paramsFile.empty()) {
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

int PV_Init::commInit(int *argc, char ***argv) {
   int mpiInit;
   // If MPI wasn't initialized, initialize it.
   // Remember if it was initialized on entry; the destructor will only finalize
   // if the constructor
   // init'ed.
   // This way, you can do several simulations sequentially by initializing MPI
   // before creating
   // the first HyPerCol; after running the first simulation the MPI environment
   // will still exist
   // and you
   // can run the second simulation, etc.
   MPI_Initialized(&mpiInit);
   if (!mpiInit) {
      pvAssert((*argv)[*argc] == nullptr); // Open MPI 1.7 assumes this.
      MPI_Init(argc, argv);
   }
   else {
      Fatal() << "PV_Init communicator already initialized\n";
   }

   return 0;
}

void PV_Init::initFactory() { PV::registerCoreKeywords(); }

void PV_Init::initLogFile(bool appendFlag) {
   // TODO: Under MPI, non-root processes should send messages to root process.
   // Currently, if logFile is directory/filename.txt, the root process writes to that path,
   // and nonroot processes write to directory/filename_<rank>.txt, where <rank> is replaced with
   // the global rank.
   // If filename does not have an extension, _<rank> is appended.
   // Note that the global rank zero process does not insert _<rank>.  This is deliberate, as the
   // nonzero ranks should be MPI-ing the data to the zero rank.
   std::string logFile         = mArguments->getStringArgument("LogFile");
   int const globalRootProcess = 0;
   int globalRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
   std::ios_base::openmode mode =
         appendFlag ? std::ios_base::out | std::ios_base::app : std::ios_base::out;
   if (!logFile.empty() && globalRank != globalRootProcess) {
      // To prevent collisions caused by multiple processes opening the same file for logging,
      // processes with global rank other than zero append the rank to the log filename.
      // If the logfile has an extension (e.g. ".log", ".txt"), the rank is appended before the
      // period separating the extension.
      std::string directory     = dirName(logFile);
      std::string stripExt      = stripExtension(logFile);
      std::string fileExt       = extension(logFile);
      std::string logFileString =
            directory + '/' + stripExt + '_' + std::to_string(globalRank) + fileExt;
      PV::setLogFile(logFileString, mode);
   }
   else {
      PV::setLogFile(logFile, mode);
   }
}

int PV_Init::setParams(char const *params_file) {
   if (params_file == nullptr) {
      return PV_FAILURE;
   }
   mArguments->setStringArgument("ParamsFile", std::string{params_file});
   initialize();
   return createParams();
}

int PV_Init::createParams() {
   std::string paramsFile = mArguments->getStringArgument("ParamsFile");
   if (!paramsFile.empty()) {
      delete params;
      params = new PVParams(
            paramsFile.c_str(),
            2 * (INITIAL_LAYER_ARRAY_SIZE + INITIAL_CONNECTION_ARRAY_SIZE),
            mCommunicator->globalCommunicator());
      unsigned int shuffleSeed = mArguments->getUnsignedIntArgument("ShuffleParamGroups");
      if (shuffleSeed) {
         params->shuffleGroups(shuffleSeed);
      }
      return PV_SUCCESS;
   }
   else {
      return PV_FAILURE;
   }
}

int PV_Init::setLogFile(char const *logFile, bool appendFlag) {
   std::string logFileString{logFile};
   mArguments->setStringArgument("LogFile", logFileString);
   initLogFile(appendFlag);
   printInitMessage();
   return PV_SUCCESS;
}

int PV_Init::setMPIConfiguration(int rows, int columns, int batchWidth) {
   if (rows >= 0) {
      mArguments->setIntegerArgument("NumRows", rows);
   }
   if (columns >= 0) {
      mArguments->setIntegerArgument("NumColumns", columns);
   }
   if (batchWidth >= 0) {
      mArguments->setIntegerArgument("BatchWidth", batchWidth);
   }
   initialize();
   return PV_SUCCESS;
}

void PV_Init::printInitMessage() {
   InfoLog() << "PID " << getpid() << ", global rank " << getWorldRank() << ".\n";
   time_t currentTime = time(nullptr);
   InfoLog() << "PetaVision initialized at "
             << ctime(&currentTime); // string returned by ctime contains a trailing \n.
   struct utsname systemInfo;
   int unamestatus = uname(&systemInfo);
   if (unamestatus == 0) {
      InfoLog() << "System information:\n"
                << "    system name: " << systemInfo.sysname << "\n"
                << "    nodename:    " << systemInfo.nodename << "\n"
                << "    release:     " << systemInfo.release << "\n"
                << "    version:     " << systemInfo.version << "\n"
                << "    machine:     " << systemInfo.machine << "\n";
   }
   else {
      ErrorLog() << "System name information unavailable: " << strerror(errno) << "\n";
   }
   auto globalMPIBlock = getCommunicator()->getGlobalMPIBlock();
   auto ioMPIBlock = getCommunicator()->getIOMPIBlock();
   InfoLog().printf("----------------\n");
   InfoLog() << "Configuration is:\n";
   printState();
   InfoLog().printf("----------------\n");
   InfoLog() << "Running with NumRows=" << mCommunicator->numCommRows()
             << ", NumCols=" << mCommunicator->numCommColumns()
             << ", and BatchWidth=" << mCommunicator->numCommBatches() << "\n";
   InfoLog() << "I/O Blocks have " << ioMPIBlock->getNumRows() << " rows, "
             << ioMPIBlock->getNumColumns() << " columns, and "
             << "batch width of " << ioMPIBlock->getBatchDimension() << "\n";
   InfoLog().printf("Position in global MPI configuration:\n");
   InfoLog().printf(
        "    Row %d of %d, Column %d of %d, Batch index %d of %d\n",
        globalMPIBlock->getRowIndex(),
        globalMPIBlock->getNumRows(),
        globalMPIBlock->getColumnIndex(),
        globalMPIBlock->getNumColumns(),
        globalMPIBlock->getBatchIndex(),
        globalMPIBlock->getBatchDimension());
   InfoLog().printf(
        "Input/Output block starts at row %d, column %d, batch index %d\n",
        ioMPIBlock->getStartRow(),
        ioMPIBlock->getStartColumn(),
        ioMPIBlock->getStartBatch());
   InfoLog().printf("Position within Input/Output block:\n");
   InfoLog().printf(
        "    Row %d of %d, Column %d of %d, Batch index %d of %d\n",
        ioMPIBlock->getRowIndex(),
        ioMPIBlock->getNumRows(),
        ioMPIBlock->getColumnIndex(),
        ioMPIBlock->getNumColumns(),
        ioMPIBlock->getBatchIndex(),
        ioMPIBlock->getBatchDimension());
}

int PV_Init::resetState() {
   mArguments->resetState();
   return PV_SUCCESS;
}

int PV_Init::registerKeyword(char const *keyword, ObjectCreateFn creator) {
   int status = Factory::instance()->registerKeyword(keyword, creator);
   if (status != PV_SUCCESS) {
      if (getWorldRank() == 0) {
         ErrorLog().printf("PV_Init: keyword \"%s\" has already been registered.\n", keyword);
      }
   }
   return status;
}

char **PV_Init::getArgsCopy() const {
   char **argumentArray = (char **)pvMallocError(
         (size_t)(mArgC + 1) * sizeof(char *),
         "PV_Init::getArgsCopy  allocate memory for %d arguments: %s\n",
         mArgC,
         strerror(errno));
   for (int a = 0; a < mArgC; a++) {
      char const *arga = mArgV[a];
      if (arga) {
         char *copied = strdup(arga);
         if (!copied) {
            ErrorLog().printf("PV_Init unable to store argument %d: %s\n", a, strerror(errno));
            Fatal().printf("Argument was \"%s\".\n", arga);
         }
         argumentArray[a] = copied;
      }
      else {
         argumentArray[a] = nullptr;
      }
   }
   argumentArray[mArgC] = nullptr;
   return argumentArray;
}

void PV_Init::freeArgs(int argc, char **argv) {
   for (int k = 0; k < argc; k++) {
      free(argv[k]);
   }
   free(argv);
}

int PV_Init::commFinalize() {
   MPI_Finalize();
   return 0;
}

} // namespace PV
