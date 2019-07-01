#include "bindings/Interactions.hpp"
#include "bindings/InteractionMessages.hpp"

namespace PV {


// Private

Response::Status Interactions::interact(std::shared_ptr<InteractionMessage const> message) {
   return mHC->interact(message);
}

void Interactions::clearError() {
   mErrMsg = "";
}

void Interactions::error(std::string const err) {
   if (mErrMsg == "") {
      mErrMsg = err;
   }
}

Interactions::Result Interactions::checkError(std::shared_ptr<InteractionMessage const> message, Response::Status status,
      std::string const funcName, std::string const objName) {
   clearError();
   if (status != Response::SUCCESS) {
      if (getError() == "") {
         error(funcName + ": failed to find object '" + objName + "'");
         return FAILURE;
      }
      error(funcName + ": " + getError());
      return FAILURE;
   }
   return SUCCESS;
}

// Public

std::string const Interactions::getError() {
   return mErrMsg;
}

Interactions::Interactions(std::map<std::string, std::string> args, std::string params) {
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

Interactions::~Interactions() {
   delete(mHC);
   delete(mPVI);
   for (int i = 0; i < mArgC; i++) {
      free(mArgV[i]);
   }
   free(mArgV);
}

Interactions::Result Interactions::begin() {
   clearError();
   if (mPVI->isExtraProc()) {
      error("Too many processes were allocated.");
      return FAILURE;
   }

   PVParams *params = mPVI->getParams();
 
   if (params == nullptr) {
      error("beginRun was called without valid params");
      return FAILURE;
   }

   // TODO: Ignoring param sweep here, fix later

   mHC = new HyPerCol(mPVI);
   mHC->startRun();

   return SUCCESS;
}

Interactions::Result Interactions::step(double *simTime) {
   double s = 0.0;
   s = mHC->singleStep(); 
   if (simTime != nullptr) {
      *simTime = s;
   }
   return SUCCESS;
}

Interactions::Result Interactions::finish() {
   mHC->finishRun();
   return SUCCESS;
}

Interactions::Result Interactions::getLayerActivity(const char *layerName, std::vector<float> *data) {
   auto message = std::make_shared<LayerGetActivityMessage>(&mErrMsg, layerName, data);
   auto status  = interact(message);
   return checkError(message, status, "getLayerActivity", std::string(layerName));
}

Interactions::Result Interactions::getLayerState(const char *layerName, std::vector<float> *data) {
   auto message = std::make_shared<LayerGetInternalStateMessage>(&mErrMsg, layerName, data);
   auto status  = interact(message);
   return checkError(message, status, "getLayerState", std::string(layerName));
}

Interactions::Result Interactions::setLayerState(const char *layerName, const std::vector<float> *data) {
   auto message = std::make_shared<LayerSetInternalStateMessage>(&mErrMsg, layerName, data);
   auto status  = interact(message);
   return checkError(message, status, "setLayerState", std::string(layerName));
}

Interactions::Result Interactions::getLayerShape(const char *layerName, PVLayerLoc *loc) {
   auto message = std::make_shared<LayerGetShapeMessage>(&mErrMsg, layerName, loc);
   auto status  = interact(message);
   return checkError(message, status, "getLayerShape", std::string(layerName));
}

Interactions::Result Interactions::getProbeValues(const char *probeName, std::vector<double> *data) {
   auto message = std::make_shared<ProbeGetValuesMessage>(&mErrMsg, probeName, data);
   auto status  = interact(message);
   return checkError(message, status, "getProbeValues", std::string(probeName));
}

Interactions::Result Interactions::getConnectionWeights(const char *connName, std::vector<float> *data) {
   auto message = std::make_shared<ConnectionGetWeightsMessage>(&mErrMsg, connName, data);
   auto status  = interact(message);
   return checkError(message, status, "getConnectionWeights", std::string(connName));
}

Interactions::Result Interactions::setConnectionWeights(const char *connName, const std::vector<float> *data) {
   auto message = std::make_shared<ConnectionSetWeightsMessage>(&mErrMsg, connName, data);
   auto status  = interact(message);
   return checkError(message, status, "setConnectionWeights", std::string(connName));
}

Interactions::Result Interactions::getConnectionPatchGeometry(const char *connName, int *nwp, int *nyp, int *nxp, int *nfp) {
   auto message = std::make_shared<ConnectionGetPatchGeometryMessage>(&mErrMsg, connName, nwp, nyp, nxp, nfp);
   auto status  = interact(message);
   return checkError(message, status, "getConnectionPatchGeometry", std::string(connName));
}

bool Interactions::isFinished() {
   return mHC->simulationTime() >= mHC->getStopTime() - mHC->getDeltaTime() / 2.0;
}

int Interactions::getMPIShape(int *rows, int *cols, int *batches) {
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

int Interactions::getMPILocation(int *row, int *col, int *batch) {
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
