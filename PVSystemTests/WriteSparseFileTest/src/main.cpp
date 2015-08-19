/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <io/RequireAllZeroActivityProbe.hpp>
#include <layers/MoviePvp.cpp>

#define CORRECT_PVP_NX 32 // The x-dimension in the "correct.pvp" file.  Needed by generate()
#define CORRECT_PVP_NY 32 // The y-dimension in the "correct.pvp" file.  Needed by generate()
#define CORRECT_PVP_NF 8 // The number of features in the "correct.pvp" file.  Needed by generate()

int copyCorrectOutput(HyPerCol * hc, int argc, char * argv[]);
int assertAllZeroes(HyPerCol * hc, int argc, char * argv[]);

int generate(int argc, char * argv[], PV_Init* initObj, int rank);
int testrun(int argc, char * argv[], PV_Init* initObj, int rank);
int testcheckpoint(int argc, char * argv[], PV_Init* initObj, int rank);
int testioparams(int argc, char * argv[], PV_Init* initObj, int rank);

int main(int argc, char * argv[]) {
   PV_Init* initObj = new PV_Init(&argc, &argv); 
   int rank = 0;
#ifdef PV_USE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif // PV_USE_MPI
   //int rank = initObj->getWorldRank();

   int pv_argc = 0;
   bool generateFlag = false; // Flag for whether to generate correct output for future tests; don't check the RequireAllZeroActivity probe
   bool testrunFlag = false;  // Flag for whether to run from params and then check the RequireAllZeroActivity probe
   bool testcheckpointFlag = false;  // Flag for whether to run from checkpoint and then check the RequireAllZeroActivity probe
   bool testioparamsFlag = false; // Flag for whether to run from the output pv.params and then check the RequireAllZeroActivity probe
   char ** pv_argv = (char **) calloc((size_t) (argc+1), sizeof(char *));
   if (pv_argv==NULL) {
      fprintf(stderr, "Error allocating memory in rank %d process: %s\n", rank, strerror(errno));
      abort();
   }

   // Run through the command line arguments.  If an argument is any of
   // --generate
   // --testrun
   // --testcheckpoint
   // --testioparams
   // --testall
   // set the appropriate test flags
   for (int arg=0; arg<argc;arg++) {
      const char * thisarg = argv[arg];
      if (!strcmp(thisarg, "--generate")) {
         generateFlag = true;
      }
      else if (!strcmp(thisarg, "--testrun")) {
         testrunFlag = true;
      }
      else if (!strcmp(thisarg, "--testcheckpoint")) {
         testcheckpointFlag = true;
      }
      else if (!strcmp(thisarg, "--testioparams")) {
         testioparamsFlag = true;
      }
      else if (!strcmp(thisarg, "--testall")) {
         testrunFlag = true;
         testcheckpointFlag = true;
         testioparamsFlag = true;
      }
      else {
         pv_argv[pv_argc] = argv[arg];
         pv_argc++;
         assert(pv_argc<=argc);
      }
   }
   pv_argv[pv_argc]=NULL;
   if (generateFlag && (testrunFlag||testcheckpointFlag||testioparamsFlag)) {
      if (rank==0) {
         fprintf(stderr, "%s error: --generate option conflicts with the --test* options.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD); // Make sure no child processes take down the MPI environment before root process prints error message.
      exit(EXIT_FAILURE);
   }
   if (!(generateFlag||testrunFlag||testcheckpointFlag||testioparamsFlag)) {
      if (rank==0) {
         fprintf(stderr, "%s error: At least one of \"--generate\", \"--testrun\", \"--testcheckpoint\", \"testioparams\" must be selected.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD); // Make sure no child processes take down the MPI environment before root process prints error message.
      exit(EXIT_FAILURE);
   }
   assert(generateFlag||testrunFlag||testcheckpointFlag||testioparamsFlag);

   int status = PV_SUCCESS;
   if (status==PV_SUCCESS && generateFlag) {
      if (generate(pv_argc, pv_argv, initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            fprintf(stderr, "%s: generate failed.\n", pv_argv[0]);
         }
      }
   }
   if (status==PV_SUCCESS && testrunFlag) {
      if (testrun(pv_argc, pv_argv, initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            fprintf(stderr, "%s: testrun failed.\n", pv_argv[0]);
         }
      }
   }
   if (status==PV_SUCCESS && testcheckpointFlag) {
      if (testcheckpoint(pv_argc, pv_argv, initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            fprintf(stderr, "%s: testcheckpoint failed.\n", pv_argv[0]);
         }
      }
   }
   if (status==PV_SUCCESS && testioparamsFlag) {
      if (testioparams(pv_argc, pv_argv, initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            fprintf(stderr, "%s: testioparams failed.\n", pv_argv[0]);
         }
      }
   }
   free(pv_argv); pv_argv = NULL;

//#ifdef PV_USE_MPI
//   MPI_Finalize();
//#endif // PV_USE_MPI
   delete initObj;

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int generate(int argc, char * argv[], PV_Init* initObj, int rank) {
   // Remove -r and -c
   char ** pv_argv = (char **) calloc((size_t) (argc+1), sizeof(char *));
   assert(pv_argv);
   int pv_argc = 0;
   for (int arg=0; arg<argc; arg++) {
      if (!strcmp(argv[arg], "-c")) {
         arg++; // skip the argument to -c
      }
      else if (!strcmp(argv[arg],"-r")) {
         // -r has no arguments
      }
      else {
         pv_argv[pv_argc] = argv[arg];
         pv_argc++;
      }
   }
   if (rank==0) {
      printf("%s --generate running PetaVision with arguments\n", pv_argv[0]);
      for (int i=1; i<pv_argc; i++) {
         printf(" %s", pv_argv[i]);
      }
      printf("\n");
   }
   if (rank==0) {
      PV_Stream * emptyinfile = PV_fopen("input/correct.pvp", "w", false/*verifyWrites*/);
      // Data for a CORRECT_PVP_NX-by-CORRECT_PVP_NY layer with CORRECT_PVP_NF features.
      // Sparse activity with no active neurons so file size doesn't change with number of features
      int emptydata[] = {80, 20, 2, CORRECT_PVP_NX, CORRECT_PVP_NY, CORRECT_PVP_NF, 1, 0, 4, 2, 1, 1, CORRECT_PVP_NX, CORRECT_PVP_NY, 0, 0, 0, 1, 0, 0, 0, 0, 0};
      size_t numwritten = PV_fwrite(emptydata, 23, sizeof(int), emptyinfile);
      if (numwritten != 23) {
         fprintf(stderr, "%s error writing placeholder data into input/correct.pvp file.\n", pv_argv[0]);
      }
      PV_fclose(emptyinfile);
   }
   int status = rebuildandrun(pv_argc, pv_argv, initObj, NULL, &copyCorrectOutput, NULL);
   return status;
}

int copyCorrectOutput(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   std::string sourcePathString = hc->getOutputPath();
   sourcePathString += "/" "a1_output.pvp";
   const char * sourcePath = sourcePathString.c_str();
   MoviePvp * correctLayer = dynamic_cast<MoviePvp *>(hc->getLayerFromName("correct"));
   assert(correctLayer);
   const char * destPath = correctLayer->getInputPath();
   if (strcmp(&destPath[strlen(destPath)-4], ".pvp")!=0) {
      if (hc->columnId()==0) {
         fprintf(stderr, "%s --generate: This system test assumes that the layer \"correct\" is a Movie layer with imageListPath ending in \".pvp\".\n", argv[0]);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (hc->columnId()==0) {
      PV_Stream * infile = PV_fopen(sourcePath, "r", false/*verifyWrites*/);
      assert(infile);
      PV_fseek(infile, 0L, SEEK_END);
      long int filelength = PV_ftell(infile);
      PV_fseek(infile, 0L, SEEK_SET);
      char * buf = (char *) malloc((size_t) filelength);
      size_t charsread = PV_fread(buf, sizeof(char), (size_t) filelength, infile);
      assert(charsread == (size_t) filelength);
      PV_fclose(infile); infile = NULL;
      PV_Stream * outfile = PV_fopen(destPath, "w", false/*verifyWrites*/);
      assert(outfile);
      size_t charswritten = PV_fwrite(buf, sizeof(char), (size_t) filelength, outfile);
      assert(charswritten == (size_t) filelength);
      PV_fclose(outfile); outfile = NULL;
      free(buf); buf = NULL;
   }
   return status;
}

int testrun(int argc, char * argv[], PV_Init * initObj, int rank) {
   // Ignore -r and -c switches
   char ** pv_argv = (char **) calloc((size_t) (argc+1), sizeof(char *));
   assert(pv_argv);
   int pv_argc = 0;
   for (int arg=0; arg<argc; arg++) {
      if (!strcmp(argv[arg], "-c")) {
         arg++; // skip the argument to -c
      }
      else if (!strcmp(argv[arg],"-r")) {
         // -r has no arguments
      }
      else {
         pv_argv[pv_argc] = argv[arg];
         pv_argc++;
      }
   }
   if (rank==0) {
      printf("%s --testrun running PetaVision with arguments\n", pv_argv[0]);
      for (int i=1; i<pv_argc; i++) {
         printf(" %s", pv_argv[i]);
      }
      printf("\n");
   }
   int status = rebuildandrun(pv_argc, pv_argv, initObj, NULL, &assertAllZeroes, NULL);
   free(pv_argv); pv_argv = NULL;
   return status;
}

int testcheckpoint(int argc, char * argv[], PV_Init * initObj, int rank) {
   // Make sure there's either a -r or a -c switch
   bool hasrestart = false;
   for (int arg=0; arg<argc; arg++) {
      if (!strcmp(argv[arg],"-r") || !strcmp(argv[arg],"-c")) {
         hasrestart = true;
         break;
      }
   }
   if (!hasrestart) {
      if (rank==0) {
         fprintf(stderr, "%s error: --testcheckpoint requires either -r or -c option.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }
   if (rank==0) {
      printf("%s --testcheckpoint running PetaVision with arguments\n", argv[0]);
      for (int i=1; i<argc; i++) {
         printf(" %s", argv[i]);
      }
      printf("\n");
   }
   int status = rebuildandrun(argc, argv, initObj, NULL, &assertAllZeroes, NULL);
   return status;
}

int testioparams(int argc, char * argv[], PV_Init* initObj, int rank) {
   // Ignore -r and -c switches
   char ** pv_argv = (char **) calloc((size_t) (argc+1), sizeof(char *));
   assert(pv_argv);
   int pv_argc = 0;
   for (int arg=0; arg<argc; arg++) {
      if (!strcmp(argv[arg], "-c")) {
         arg++; // skip the argument to -c
      }
      else if (!strcmp(argv[arg],"-r")) {
         // -r has no arguments
      }
      else {
         pv_argv[pv_argc] = argv[arg];
         pv_argc++;
      }
   }
   initObj->initialize(pv_argc, pv_argv);
   HyPerCol * hc = build(pv_argc, pv_argv, initObj);
   if (hc == NULL) {
      fprintf(stderr, "testioparams error: unable to build HyPerCol.\n");
      exit(EXIT_FAILURE);
   }
   int status = hc->run(); // Needed to generate pv.params file
   if (status != PV_SUCCESS) {
      fprintf(stderr, "testioparams error: run to generate pv.params file failed.\n");
      exit(EXIT_FAILURE);
   }
   const char * paramsfile = hc->getPrintParamsFilename();
   std::string paramsfileString = paramsfile;
   if (paramsfile[0]!='/') {
      const char * outputPath = hc->getOutputPath();
      paramsfileString.insert(0, "/");
      paramsfileString.insert(0, outputPath);
   }
   delete hc;

   int arg; // arg will store the index where the new params file is stored in pv_argv.  Since we're using strdup, we have to free it.
   for (arg=0; arg<pv_argc; arg++) {
      if (!strcmp(pv_argv[arg], "-p")) {
         arg++;
         pv_argv[arg] = strdup(paramsfileString.c_str());
         break;
      }
   }

   bool usingdefaultparamsfile = false;
   if (arg>=pv_argc) {
      usingdefaultparamsfile = true;
      char ** new_pv_argv = (char **) malloc((size_t) (pv_argc+3)*sizeof(char *));
      assert(new_pv_argv);
      for (int argn=0; argn<pv_argc; argn++) {
         new_pv_argv[argn] = pv_argv[argn];
      }
      new_pv_argv[pv_argc] = strdup("-p");
      new_pv_argv[pv_argc+1] = strdup(paramsfileString.c_str());
      new_pv_argv[pv_argc+2] = NULL;
      arg = pv_argc+1;
      pv_argc += 2;
      free(pv_argv);
      pv_argv = new_pv_argv;
   }
   if (rank==0) {
      printf("%s --testioparams running PetaVision with arguments\n", pv_argv[0]);
      for (int i=1; i<pv_argc; i++) {
         printf(" %s", pv_argv[i]);
      }
      printf("\n");
   }
   status = rebuildandrun(pv_argc, pv_argv, initObj, NULL, &assertAllZeroes, NULL);
   if (usingdefaultparamsfile) {
      free(pv_argv[arg-1]); pv_argv[arg-1] = NULL;
   }
   free(pv_argv[arg]); pv_argv[arg] = NULL;
   free(pv_argv); pv_argv = NULL;
   return status;
}

int assertAllZeroes(HyPerCol * hc, int argc, char * argv[]) {
   const char * targetLayerName = "comparison";
   HyPerLayer * layer = hc->getLayerFromName(targetLayerName);
   LayerProbe * probe = NULL;
   int np = layer->getNumProbes();
   for (int p=0; p<np; p++) {
      if (!strcmp(layer->getProbe(p)->getName(), "comparison_test")) {
         probe = layer->getProbe(p);
         break;
      }
   }
   RequireAllZeroActivityProbe * allzeroProbe = dynamic_cast<RequireAllZeroActivityProbe *>(probe);
   assert(allzeroProbe);
   if (allzeroProbe->getNonzeroFound()) {
      if (hc->columnId()==0) {
         double t = allzeroProbe->getNonzeroTime();
         fprintf(stderr, "%s \"%s\" had at least one nonzero activity value, beginning at time %f\n",
               hc->parameters()->groupKeywordFromName(targetLayerName), targetLayerName, t);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}
