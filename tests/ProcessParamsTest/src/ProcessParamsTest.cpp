/*
 * ProcessParamsTest.cpp
 *
 */

#include <sys/types.h>
#include <unistd.h>
#include <columns/buildandrun.hpp>

#define PROCESSED_PARAMS "processed.params"

int deleteGeneratedFiles(PV::PV_Init * pv_obj);
int compareOutputs();

int deleteFile(char const * path, PV::PV_Init * pv_obj);

int main(int argc, char * argv[]) {

   int status = PV_SUCCESS;

   PV::PV_Init pv_obj(&argc, &argv, false/*allowUnrecognizedArguments*/);

   if (pv_obj.getParamsFile()==NULL) {
      pv_obj.setParams("input/ProcessParamsTest.params");
   }

   if (status != PV_SUCCESS) {
      pvError().printf("%s: PV_Init::initialize() failed on process with PID=%d\n", argv[0], getpid()); 
   }

   if (pv_obj.isExtraProc()) { return EXIT_SUCCESS; }

   int rank = pv_obj.getCommunicator()->globalCommRank();

   status = deleteGeneratedFiles(&pv_obj);
   if (status!=PV_SUCCESS) {
      pvError().printf("%s: error cleaning generated files from any previous run.\n", argv[0]);
   }

   PV::HyPerCol * hc = build(&pv_obj);
   if (hc==NULL) {
      pvError().printf("%s: build() failed on process %d\n", argv[0], rank);
   }

   // Generate the cleaned-up params file
   status = hc->processParams(PROCESSED_PARAMS);
   if (status != PV_SUCCESS) {
      pvError().printf("%s: HyPerCol::processParams failed on process %d\n", argv[0], rank);
   }

   // Run the column with the raw params file, sending output to directory "output-generate/"
   pv_obj.setOutputPath("output-generate");
   status = rebuildandrun(&pv_obj);
   if (status != PV_SUCCESS) {
      pvError().printf("%s: running with raw params file failed on process %d\n", argv[0], rank);
   }

   // Run the column with the cleaned-up params file, sending output to directory "output-verify/"
   pv_obj.setOutputPath("output-verify");
   pv_obj.setParams(PROCESSED_PARAMS);
   status = rebuildandrun(&pv_obj);
   if (status != PV_SUCCESS) {
      pvError().printf("%s: running with processed params file failed on process %d\n", argv[0], rank);
   }

   if (rank==0) {
      status = compareOutputs();
      if (status != PV_SUCCESS) {
         pvError().printf("%s: compareOutputs() failed with return code %d.\n", argv[0], status);
      }
   }

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int deleteGeneratedFiles(PV::PV_Init * pv_obj) {

   int status = PV_SUCCESS;
   if (pv_obj->getCommunicator()->globalCommRank()==0) {
      char const * filename = NULL;

      if (deleteFile(PROCESSED_PARAMS, pv_obj) != PV_SUCCESS) { status = PV_FAILURE; }
      if (deleteFile(PROCESSED_PARAMS ".lua", pv_obj) != PV_SUCCESS) { status = PV_FAILURE; }
      if (system("rm -rf output-generate") != PV_SUCCESS) { status = PV_FAILURE; }
      if (system("rm -rf output-verify") != PV_SUCCESS) { status = PV_FAILURE; }
   }
   MPI_Bcast(&status, 1, MPI_INT, 0, pv_obj->getCommunicator()->communicator());
   return status;
}

int compareOutputs() {
   // Only root process calls this function, since it does I/O
#ifndef NDEBUG
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   pvErrorIf(!(rank==0), "Test failed.\n");
#endif // NDEBUG

   int status = PV_SUCCESS;
   char const * diffcmd = "diff -r -q -x pv.params -x pv.params.lua -x timers.txt output-generate output-verify";
   status = system(diffcmd);
   if (status != 0) {
      // If the diff command fails, it may be only that the file system hasn't caught up yet.  
      pvWarn().printf("diff command returned %d: waiting 1 second and trying again...\n", WEXITSTATUS(status));
      pvWarn().flush();
      sleep(1);
      status = system(diffcmd);
      if (status!=0) { status = WEXITSTATUS(status); }
   }
   return status;
}

int deleteFile(char const * path, PV::PV_Init * pv_obj) {
   // Only root process calls this function, since it does I/O
#ifndef NDEBUG
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   pvErrorIf(!(rank==0), "Test failed.\n");
#endif // NDEBUG

   int status = unlink(path);
   if (status != 0 && errno != ENOENT) {
      pvErrorNoExit().printf("%s: unable to delete %s: %s\n", pv_obj->getProgramName(), path, strerror(errno));
      status = PV_FAILURE;
   }
   else {
      status = PV_SUCCESS;
   }
   return PV_SUCCESS;
}
