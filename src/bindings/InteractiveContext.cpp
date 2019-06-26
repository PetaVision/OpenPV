#include <bindings/InteractiveContext.hpp>
#include <columns/Messages.hpp>


namespace PV {

void InteractiveContext::message(std::shared_ptr<BaseMessage const> message) {
   mHC->externalMessage(message);
}

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
   if (args.find("Restart") != args.end() && args["Restart"] == "True") {
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
   if (args.find("DryRun") != args.end() && args["DryRun"] == "True") {
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

   mMPIRows    = mPVI->getCommunicator()->numCommRows();
   mMPICols    = mPVI->getCommunicator()->numCommColumns();
   mMPIBatches = mPVI->getCommunicator()->numCommBatches();

   mRow = mPVI->getCommunicator()->commRow();
   mCol = mPVI->getCommunicator()->commColumn();
   mBatch = mPVI->getCommunicator()->commBatch();
   mRank = mPVI->getCommunicator()->globalCommRank();

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
   FatalIf(mPVI->isExtraProc(), "Too many processes were allocated. Exiting.\n");
   PVParams *params = mPVI->getParams();
   FatalIf(params == nullptr, "beginRun was called without valid params\n");

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

void InteractiveContext::getLayerActivity(const char *layerName, std::vector<float> *data) {
   message(std::make_shared<LayerGetActivityMessage>(layerName, data));
}

void InteractiveContext::getLayerState(const char *layerName, std::vector<float> *data) {
   message(std::make_shared<LayerGetInternalStateMessage>(layerName, data));
}

void InteractiveContext::setLayerState(const char *layerName, std::vector<float> *data) {
   message(std::make_shared<LayerSetInternalStateMessage>(layerName, data));
}

void InteractiveContext::getLayerShape(const char *layerName, PVLayerLoc *loc) {
   message(std::make_shared<LayerGetShapeMessage>(layerName, loc));
}

bool InteractiveContext::isFinished() {
   return mHC->simulationTime() >= mHC->getStopTime() - mHC->getDeltaTime() / 2.0;
}

void InteractiveContext::getProbeValues(const char *probeName, std::vector<double> *data) {
   message(std::make_shared<ProbeGetValuesMessage>(probeName, data));
}

int InteractiveContext::getMPIShape(int *rows, int *cols, int *batches) {
   if (rows != NULL) {
      *rows = mMPIRows;
   }
   if (cols != NULL) {
      *cols = mMPICols;
   }
   if (batches != NULL) {
      *batches = mMPIBatches; 
   }
   return mMPIRows * mMPICols * mMPIBatches;
}

int InteractiveContext::getMPILocation(int *row, int *col, int *batch) {
   if (row != NULL) {
      *row = mRow;
   }
   if (col != NULL) {
      *col = mCol;
   }
   if (batch != NULL) {
      *batch = mBatch; 
   }
   return mRank;
}


} /* namespace PV */
