/*
 * pv.cpp
 *
 */

#include <iostream>
#include <sstream>
#include <arch/mpi/mpi.h>
#include <columns/buildandrun.hpp>

#undef MAIN_USES_CUSTOMGROUPS

int customexit(HyPerCol * hc, int argc, char * argv[]);

int main(int argc, char * argv[]) {
   const char * paramfile1 = "input/RandStateSystemTest1.params";
   const char * paramfile2 = "input/RandStateSystemTest2.params";

   int rank=0;
   PV_Init* initObj = new PV_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   if (pv_getopt(argc, argv, "-p", NULL)==0) {
      if (rank==0) {
         fprintf(stderr, "%s does not take -p as an option.  Instead the necessary params files are hard-coded.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }

   int cl_argc = argc + 2;
   char ** cl_argv = (char **) calloc(cl_argc+1,sizeof(char *));
   assert(cl_argv);
   for (int k=0; k<argc; k++) {
      cl_argv[k] = strdup(argv[k]);
      assert(cl_argv[k]);
   }
   int paramfile_argnum = argc+1;
   cl_argv[paramfile_argnum-1] = strdup("-p");
   cl_argv[paramfile_argnum] = strdup(paramfile1);
   cl_argv[paramfile_argnum+1] = NULL;

   int status1 = rebuildandrun(cl_argc, cl_argv, initObj, NULL, NULL, NULL);
   if (status1 != PV_SUCCESS) {
      fprintf(stderr, "%s failed on param file %s with return code %d.\n", cl_argv[0], cl_argv[2], status1);
      return EXIT_FAILURE;
   }

   free(cl_argv[paramfile_argnum]);
   cl_argv[paramfile_argnum] = strdup(paramfile2);
   int status2 = rebuildandrun(cl_argc, cl_argv, initObj, NULL, &customexit, NULL);
   if (status2 != PV_SUCCESS) {
      fprintf(stderr, "%s failed on param file %s.\n", cl_argv[0], cl_argv[paramfile_argnum]);
   }
   int status = status1==PV_SUCCESS && status2==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;

#ifdef PV_USE_MPI
   if (status == EXIT_SUCCESS) {
      printf("Test complete.  %s passed on process rank %d.\n", cl_argv[0], rank);
   }
   else {
      fprintf(stderr, "Test complete.  %s FAILED on process rank %d.\n", cl_argv[0], rank);
   }
#else
   if (status == EXIT_SUCCESS) {
      printf("Test complete.  %s passed.\n", cl_argv[0]);
   }
   else {
      fprintf(stderr, "Test complete.  %s FAILED.\n", cl_argv[0]);
   }
#endif // PV_USE_MPI

   for (int k=0; k<argc; k++) {
      free(cl_argv[k]);
   }
   free(cl_argv[paramfile_argnum-1]);
   free(cl_argv[paramfile_argnum]);
   free(cl_argv);

   delete initObj;

   return status;
}

int customexit(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   if (hc->columnId()==0) {
      std::string cmd("diff -r -q -x timers.txt -x pv.params -x pv.params.lua checkpoints*/Checkpoint");
      std::stringstream stepnumber;
      stepnumber << hc->getFinalStep();
      cmd += stepnumber.str();
      const char * cmdstr = cmd.c_str();
      status = system(cmdstr);
      if (status != 0) {
         fprintf(stderr, "%s failed: system command \"%s\" returned %d\n", argv[0], cmdstr, status);
      }
   }
   return status;
}
