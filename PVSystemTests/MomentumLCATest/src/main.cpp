/*
 * pv.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <io/RequireAllZeroActivityProbe.hpp>
#include <layers/MoviePvp.hpp>

#define CORRECT_PVP_NX 1440 // The x-dimension in the "correct.pvp" file.  Needed by generate()
#define CORRECT_PVP_NY 960 // The y-dimension in the "correct.pvp" file.  Needed by generate()
#define CORRECT_PVP_NF 1 // The number of features in the "correct.pvp" file.  Needed by generate()

int copyCorrectOutput(HyPerCol * hc, int argc, char * argv[]);
int assertAllZeroes(HyPerCol * hc, int argc, char * argv[]);

int generate(PV_Init* initObj, int rank);
int testrun(PV_Init* initObj, int rank);
int testcheckpoint(PV_Init* initObj, int rank);
int testioparams(PV_Init* initObj, int rank);

int main(int argc, char * argv[]) {
   PV_Init initObj(&argc, &argv, true/*allowUnrecognizedArguments*/);
   // argv has to allow --generate, --testrun, etc.
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   bool generateFlag = false; // Flag for whether to generate correct output for future tests; don't check the RequireAllZeroActivity probe
   bool testrunFlag = false;  // Flag for whether to run from params and then check the RequireAllZeroActivity probe
   bool testcheckpointFlag = false;  // Flag for whether to run from checkpoint and then check the RequireAllZeroActivity probe
   bool testioparamsFlag = false; // Flag for whether to run from the output pv.params and then check the RequireAllZeroActivity probe

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
         /* nothing to do here */
      }
   }
   if (generateFlag && (testrunFlag||testcheckpointFlag||testioparamsFlag)) {
      if (rank==0) {
         fprintf(stderr, "%s error: --generate option conflicts with the --test* options.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD); // Make sure no child processes take down the MPI environment before root process prints error message.
      exit(EXIT_FAILURE);
   }
   if (!(generateFlag||testrunFlag||testcheckpointFlag||testioparamsFlag)) {
      if (rank==0) {
         fprintf(stderr, "%s error: At least one of \"--generate\", \"--testrun\", \"--testcheckpoint\", \"--testioparams\" must be selected.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD); // Make sure no child processes take down the MPI environment before root process prints error message.
      exit(EXIT_FAILURE);
   }
   assert(generateFlag||testrunFlag||testcheckpointFlag||testioparamsFlag);

   int status = PV_SUCCESS;
   if (status==PV_SUCCESS && generateFlag) {
      if (generate(&initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            fprintf(stderr, "%s: generate failed.\n", initObj.getArguments()->getProgramName());
         }
      }
   }
   if (status==PV_SUCCESS && testrunFlag) {
      if (testrun(&initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            fprintf(stderr, "%s: testrun failed.\n", initObj.getArguments()->getProgramName());
         }
      }
   }
   if (status==PV_SUCCESS && testcheckpointFlag) {
      if (testcheckpoint(&initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            fprintf(stderr, "%s: testcheckpoint failed.\n", initObj.getArguments()->getProgramName());
         }
      }
   }
   if (status==PV_SUCCESS && testioparamsFlag) {
      if (testioparams(&initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            fprintf(stderr, "%s: testioparams failed.\n", initObj.getArguments()->getProgramName());
         }
      }
   }

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int generate(PV_Init* initObj, int rank) {
   PV_Arguments * arguments = initObj->getArguments();

   // Remove -r and -c
   arguments->setRestartFlag(false);
   arguments->setCheckpointReadDir(NULL);
   if (rank==0) {
      printf("%s --generate running PetaVision with arguments\n", arguments->getProgramName());
      arguments->printState();
   }
   if (rank==0) {
      PV_Stream * emptyinfile = PV_fopen("input/correct.pvp", "w", false/*verifyWrites*/);
      // Data for a CORRECT_PVP_NX-by-CORRECT_PVP_NY layer with CORRECT_PVP_NF features.
      // Sparse activity with no active neurons so file size doesn't change with number of features
      int emptydata[] = {80, 20, 2, CORRECT_PVP_NX, CORRECT_PVP_NY, CORRECT_PVP_NF, 1, 0, 4, 2, 1, 1, CORRECT_PVP_NX, CORRECT_PVP_NY, 0, 0, 0, 1, 0, 0, 0, 0, 0};
      size_t numwritten = PV_fwrite(emptydata, 23, sizeof(int), emptyinfile);
      if (numwritten != 23) {
         fprintf(stderr, "%s error writing placeholder data into input/correct.pvp file.\n", arguments->getProgramName());
      }
      PV_fclose(emptyinfile);
   }
   int status = rebuildandrun(initObj, NULL, &copyCorrectOutput);
   return status;
}

int copyCorrectOutput(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   std::string sourcePathString = hc->getOutputPath();
   sourcePathString += "/" "a3_Reconstruction.pvp";
   const char * sourcePath = sourcePathString.c_str();
   MoviePvp * correctLayer = dynamic_cast<MoviePvp *>(hc->getLayerFromName("Correct"));
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

int testrun(PV_Init * initObj, int rank) {
   PV_Arguments * arguments = initObj->getArguments();
   arguments->resetState();
   // Ignore restart flag and checkpoint directory
   arguments->setRestartFlag(false);
   arguments->setCheckpointReadDir(NULL);
   if (rank==0) {
      printf("%s --testrun running PetaVision with arguments\n", arguments->getProgramName());
      arguments->printState();
   }
   int status = rebuildandrun(initObj, NULL, &assertAllZeroes);
   return status;
}

int testcheckpoint(PV_Init * initObj, int rank) {
   PV_Arguments * arguments = initObj->getArguments();
   arguments->resetState();
   // Make sure either restartFlag or checkpointReadDir are set (both cannot be set or PV_Arguments will error out).
   bool hasrestart = (arguments->getRestartFlag() || arguments->getCheckpointReadDir()!=NULL);
   if (!hasrestart) {
      if (rank==0) {
         fprintf(stderr, "%s error: --testcheckpoint requires either the -r or the -c option.\n", arguments->getProgramName());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }
   if (rank==0) {
      printf("%s --testcheckpoint running PetaVision with arguments\n", arguments->getProgramName());
      arguments->printState();
   }
   int status = rebuildandrun(initObj, NULL, &assertAllZeroes);
   return status;
}

int testioparams(PV_Init* initObj, int rank) {
   PV_Arguments * arguments = initObj->getArguments();
   arguments->resetState();
   // Ignore -r and -c switches
   arguments->setRestartFlag(false);
   arguments->setCheckpointReadDir(NULL);
   initObj->initialize();
   HyPerCol * hc = build(initObj);
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

   arguments->setParamsFile(paramsfileString.c_str());
   if (rank==0) {
      printf("%s --testioparams running PetaVision with arguments\n", arguments->getProgramName());
      arguments->printState();
   }
   status = rebuildandrun(initObj, NULL, &assertAllZeroes);
   return status;
}

int assertAllZeroes(HyPerCol * hc, int argc, char * argv[]) {
   const char * targetLayerName = "Comparison";
   HyPerLayer * layer = hc->getLayerFromName(targetLayerName);
   assert(layer);
   LayerProbe * probe = NULL;
   int np = layer->getNumProbes();
   for (int p=0; p<np; p++) {
      if (!strcmp(layer->getProbe(p)->getName(), "ComparisonTest")) {
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
               layer->getKeyword(), targetLayerName, t);
      }
      MPI_Barrier(hc->icCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}
