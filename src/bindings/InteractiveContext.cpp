#include <bindings/InteractiveContext.hpp>
#include <columns/Messages.hpp>


namespace PV {

InteractiveContext::InteractiveContext(std::map<std::string, std::string> args, std::string params) {
   std::vector<std::string> cliArgs;

   cliArgs.push_back("OpenPV");

   if (args.find("OutputPath") != args.end()) {
      cliArgs.push_back("-o");
      cliArgs.push_back(args["OutputPath"]);
   }
   if (args.find("ParamsFile") != args.end()) {
      cliArgs.push_back("-p");
      cliArgs.push_back(args["ParamsFile"]);
   }
   if (args.find("LogFile") != args.end()) {
      cliArgs.push_back("-l");
      cliArgs.push_back(args["LogFile"]);
   }
   if (args.find("GPUDevices") != args.end()) {
      cliArgs.push_back("-d");
      cliArgs.push_back(args["GPUDevices"]);
   }
   if (args.find("RandomSeed") != args.end()) {
      cliArgs.push_back("-s");
      cliArgs.push_back(args["RandomSeed"]);
   }
   if (args.find("WorkingDirectory") != args.end()) {
      cliArgs.push_back("-w");
      cliArgs.push_back(args["WorkingDirectory"]);
   }
   if (args.find("Restart") != args.end()) {
      cliArgs.push_back("-r");
   }
   if (args.find("CheckpointReadDirectory") != args.end()) {
      cliArgs.push_back("-c");
      cliArgs.push_back(args["CheckpointReadDirectory"]);
   }
   if (args.find("NumThreads") != args.end()) {
      cliArgs.push_back("-t");
      cliArgs.push_back(args["NumThreads"]);
   }
   if (args.find("BatchWidth") != args.end()) {
      cliArgs.push_back("-batchwidth");
      cliArgs.push_back(args["BatchWidth"]);
   }
   if (args.find("NumRows") != args.end()) {
      cliArgs.push_back("-rows");
      cliArgs.push_back(args["NumRows"]);
   }
   if (args.find("NumColumns") != args.end()) {
      cliArgs.push_back("-columns");
      cliArgs.push_back(args["NumColumns"]);
   }
   if (args.find("DryRun") != args.end()) {
      cliArgs.push_back("-n");
   }
   if (args.find("Shuffle") != args.end()) {
      cliArgs.push_back("-shuffle");
      cliArgs.push_back(args["Shuffle"]);
   }

   // Build argc and argv for PV_Init
   mArgC = cliArgs.size();
   mArgV = (char**)calloc(mArgC + 1, sizeof(char*));
   mArgV[mArgC] = NULL;
   for (int i = 0; i < mArgC; i++) {
      mArgV[i] = (char*)calloc(cliArgs[i].length()+1, sizeof(char));
      strcpy(mArgV[i], cliArgs[i].c_str());
   }

   mPVI = new PV_Init(&mArgC, &mArgV, false);

   // Read params from a string instead of a file
   if (!params.empty()) {
      mPVI->setParamsBuffer(params.c_str(), params.length());
   }
}

InteractiveContext::~InteractiveContext() {
   delete(mHC);
   delete(mPVI);
   for (int i = 0; i < mArgC; i++) {
      free(mArgV[i]);
   }
   free(mArgV);
}

void InteractiveContext::beginRun() {
   if (mPVI->isExtraProc()) {
      return; //TODO: Some sort of feedback?
   }
   PVParams *params = mPVI->getParams();
   if (params == NULL) {
      if (mPVI->getWorldRank() == 0) {
         char const *progName = mPVI->getProgramName();
         if (progName == NULL) {
            progName = "PetaVision";
         }
         ErrorLog().printf("%s was called without having set a params file\n", progName);
      }
      MPI_Barrier(mPVI->getCommunicator()->communicator());
      exit(EXIT_FAILURE);
   }
   // TODO: Ignoring param sweep here, fix later

   mHC = new HyPerCol(mPVI);
   mHC->startRun();
}

double InteractiveContext::advanceRun(unsigned int steps) {
   return mHC->multiStep(steps); 
}

void InteractiveContext::finishRun() {
   mHC->finishRun();
}

void InteractiveContext::getLayerActivity(const char *layerName, std::vector<float> *data,
                  int *nx, int *ny, int *nf) {
   mHC->externalMessage(std::make_shared<LayerGetActivityMessage>(layerName, data, nx, ny, nf));
}

void InteractiveContext::getLayerState(const char *layerName, std::vector<float> *data,
                  int *nx, int *ny, int *nf) {
   mHC->externalMessage(std::make_shared<LayerGetInternalStateMessage>(layerName, data, nx, ny, nf));
}


void InteractiveContext::setLayerState(const char *layerName, std::vector<float> *data) {
   mHC->externalMessage(std::make_shared<LayerSetInternalStateMessage>(layerName, data));
}

} /* namespace PV */
