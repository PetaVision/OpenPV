/*
 * main.cpp
 *
 */


#include <columns/buildandrun.hpp>
#include <io/RequireAllZeroActivityProbe.hpp>
#include <layers/InputLayer.hpp>

#define CORRECT_PVP_NX 128 // The x-dimension in the "correct.pvp" file.  Needed by generate()
#define CORRECT_PVP_NY 64 // The y-dimension in the "correct.pvp" file.  Needed by generate()
#define CORRECT_PVP_NF 3 // The number of features in the "correct.pvp" file.  Needed by generate()

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
         pvErrorNoExit().printf("%s: --generate option conflicts with the --test* options.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD); // Make sure no child processes take down the MPI environment before root process prints error message.
      exit(EXIT_FAILURE);
   }
   if (!(generateFlag||testrunFlag||testcheckpointFlag||testioparamsFlag)) {
      if (rank==0) {
         pvErrorNoExit().printf("%s: At least one of \"--generate\", \"--testrun\", \"--testcheckpoint\", \"--testioparams\" must be selected.\n", argv[0]);
      }
      MPI_Barrier(MPI_COMM_WORLD); // Make sure no child processes take down the MPI environment before root process prints error message.
      exit(EXIT_FAILURE);
   }
   pvErrorIf(!(generateFlag||testrunFlag||testcheckpointFlag||testioparamsFlag), "Test failed.\n");

   int status = PV_SUCCESS;
   if (status==PV_SUCCESS && generateFlag) {
      if (generate(&initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            pvErrorNoExit().printf("%s: generate failed.\n", initObj.getProgramName());
         }
      }
   }
   if (status==PV_SUCCESS && testrunFlag) {
      if (testrun(&initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            pvErrorNoExit().printf("%s: testrun failed.\n", initObj.getProgramName());
         }
      }
   }
   if (status==PV_SUCCESS && testcheckpointFlag) {
      if (testcheckpoint(&initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            pvErrorNoExit().printf("%s: testcheckpoint failed.\n", initObj.getProgramName());
         }
      }
   }
   if (status==PV_SUCCESS && testioparamsFlag) {
      if (testioparams(&initObj, rank)!=PV_SUCCESS) {
         status = PV_FAILURE;
         if (rank==0) {
            pvErrorNoExit().printf("%s: testioparams failed.\n", initObj.getProgramName());
         }
      }
   }

   return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

int generate(PV_Init* initObj, int rank) {
   // Remove -r and -c
   initObj->setRestartFlag(false);
   initObj->setCheckpointReadDir(NULL);
   if (rank==0) {
      pvInfo().printf("Running --generate.  Effective command line is\n");
      initObj->printState();
   }
   if (rank==0) {
      PV_Stream * emptyinfile = PV_fopen("input/correct.pvp", "w", false/*verifyWrites*/);
      // Data for a CORRECT_PVP_NX-by-CORRECT_PVP_NY layer with CORRECT_PVP_NF features.
      // Sparse activity with no active neurons so file size doesn't change with number of features
      int emptydata[] = {
            80/*header size in chars*/,
            20/*header size in 4-byte integers*/,
            2/*file type (2=sparse binary)*/,
            CORRECT_PVP_NX/*the number of pixels in the x-direction*/,
            CORRECT_PVP_NY/*the number of pixels in the y-direction*/,
            CORRECT_PVP_NF/*the number of features*/,
            1/*number of records*/,
            0/*record size*/,
            4/*data size*/,
            2/*data type*/,
            1/*number of processes in x-direction (always 1 for layers)*/,
            1/*number of processes in y-direction (always 1 for layers)*/,
            CORRECT_PVP_NX/*NX_GLOBAL (always the same as NX for layers)*/,
            CORRECT_PVP_NY/*NY_GLOBAL (always the same as NY for layers)*/,
            0/*kx0 (always 0 for layers)*/,
            0/*ky0 (always 0 for layers)*/,
            0/*number of boundary pixels*/,
            1/*number of frames*/,
            0/*first four bytes of timestamp in header(a double-precision floating-point)*/,
            0/*last four bytes of timestamp in header*/,
            0/*first four bytes of timestamp in first frame*/,
            0/*last four bytes of timestamp in first frame*/,
            0/*Number of active values in the frame (zero for empty data)*/};

      size_t numwritten = PV_fwrite(emptydata, 23, sizeof(int), emptyinfile);
      if (numwritten != 23) {
         pvErrorNoExit().printf("%s failure to write placeholder data into input/correct.pvp file.\n", initObj->getProgramName());
      }
      PV_fclose(emptyinfile);
   }
   int status = rebuildandrun(initObj, NULL, &copyCorrectOutput);
   return status;
}

int copyCorrectOutput(HyPerCol * hc, int argc, char * argv[]) {
   int status = PV_SUCCESS;
   std::string sourcePathString = hc->getOutputPath();
   sourcePathString += "/" "Reconstruction.pvp";
   const char * sourcePath = sourcePathString.c_str();
   InputLayer * correctLayer = dynamic_cast<InputLayer *>(hc->getLayerFromName("Correct"));
   pvErrorIf(!(correctLayer), "No MoviePvp layer named \"Correct\" in params.\n");
   const char * destPath = correctLayer->getInputPath().c_str();

   if (strcmp(&destPath[strlen(destPath)-4], ".pvp")!=0) {
      if (hc->columnId()==0) {
         pvErrorNoExit().printf("%s --generate: This system test assumes that the layer \"correct\" is a Movie layer with imageListPath ending in \".pvp\".\n", argv[0]);
      }
      MPI_Barrier(hc->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   if (hc->columnId()==0) {
      PV_Stream * infile = PV_fopen(sourcePath, "r", false/*verifyWrites*/);
      pvErrorIf(!infile, "Unable to open \"%s\" for reading.\n", sourcePath);
      PV_fseek(infile, 0L, SEEK_END);
      long int filelength = PV_ftell(infile);
      PV_fseek(infile, 0L, SEEK_SET);
      char * buf = (char *) malloc((size_t) filelength);
      size_t charsread = PV_fread(buf, sizeof(char), (size_t) filelength, infile);
      pvErrorIf(charsread != (size_t) filelength, "Expected to read %lu characters; only read %zu.\n", filelength, charsread);
      PV_fclose(infile); infile = NULL;
      PV_Stream * outfile = PV_fopen(destPath, "w", false/*verifyWrites*/);
      pvErrorIf(!outfile, "Unable to open \"%s\" for writing.\n", destPath);
      size_t charswritten = PV_fwrite(buf, sizeof(char), (size_t) filelength, outfile);
      pvErrorIf(charswritten != (size_t) filelength, "Attempted to write %lu characters; only wrote %zu.\n", filelength, charswritten);
      PV_fclose(outfile); outfile = NULL;
      free(buf); buf = NULL;
   }
   return status;
}

int testrun(PV_Init * initObj, int rank) {
   initObj->resetState();
   // Ignore restart flag and checkpoint directory
   initObj->setRestartFlag(false);
   initObj->setCheckpointReadDir(NULL);
   if (rank==0) {
      pvInfo().printf("Running --testrun.  Effective command line is\n");
      initObj->printState();
   }
   int status = rebuildandrun(initObj, NULL, &assertAllZeroes);
   return status;
}

int testcheckpoint(PV_Init * initObj, int rank) {
   initObj->resetState();
   // Make sure either restartFlag or checkpointReadDir are set (both cannot be set or PV_Arguments will error out).
   bool hasrestart = (initObj->getRestartFlag() || initObj->getCheckpointReadDir()!=NULL);
   if (!hasrestart) {
      if (rank==0) {
         pvErrorNoExit().printf("%s: --testcheckpoint requires either the -r or the -c option.\n", initObj->getProgramName());
      }
      MPI_Barrier(MPI_COMM_WORLD);
      exit(EXIT_FAILURE);
   }
   if (rank==0) {
      pvInfo().printf("Running --testcheckpoint.  Effective command line is\n");
   }
   int status = rebuildandrun(initObj, NULL, &assertAllZeroes);
   return status;
}

int testioparams(PV_Init* initObj, int rank) {
   initObj->resetState();
   // Ignore -r and -c switches
   initObj->setRestartFlag(false);
   initObj->setCheckpointReadDir(NULL);
   HyPerCol * hc = build(initObj);
   if (hc == NULL) {
      pvError().printf("testioparams error: unable to build HyPerCol.\n");
   }
   int status = hc->run(); // Needed to generate pv.params file
   if (status != PV_SUCCESS) {
      pvError().printf("testioparams error: run to generate pv.params file failed.\n");
   }
   const char * paramsfile = hc->getPrintParamsFilename();
   std::string paramsfileString = paramsfile;
   if (paramsfile[0]!='/') {
      const char * outputPath = hc->getOutputPath();
      paramsfileString.insert(0, "/");
      paramsfileString.insert(0, outputPath);
   }
   delete hc;

   initObj->setParams(paramsfileString.c_str());
   if (rank==0) {
      pvInfo().printf("Running --testioparams.  Effective command line is\n");
      initObj->printState();

   }
   status = rebuildandrun(initObj, NULL, &assertAllZeroes);
   return status;
}

int assertAllZeroes(HyPerCol * hc, int argc, char * argv[]) {
   const char * targetLayerName = "Comparison";
   HyPerLayer * layer = hc->getLayerFromName(targetLayerName);
   pvErrorIf(!(layer), "No layer named \"Comparison\".\n");
   LayerProbe * probe = NULL;
   int np = layer->getNumProbes();
   for (int p=0; p<np; p++) {
      if (!strcmp(layer->getProbe(p)->getName(), "ComparisonTest")) {
         probe = layer->getProbe(p);
         break;
      }
   }
   RequireAllZeroActivityProbe * allzeroProbe = dynamic_cast<RequireAllZeroActivityProbe *>(probe);
   pvErrorIf(!(allzeroProbe), "No RequireAllZeroActivityProbe named \"ComparisonTest\".\n");
   if (allzeroProbe->getNonzeroFound()) {
      if (hc->columnId()==0) {
         double t = allzeroProbe->getNonzeroTime();
         pvErrorNoExit().printf("%s had at least one nonzero activity value, beginning at time %f\n",
               layer->getDescription_c(), t);
      }
      MPI_Barrier(hc->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}
