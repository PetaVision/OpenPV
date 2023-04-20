#include <arch/mpi/mpi.h>
#include <columns/PV_Init.hpp>
#include <columns/Random.hpp>
#include <columns/RandomSeed.hpp>
#include <include/pv_common.h>
#include <io/PVParams.hpp>
#include <probes/CheckStatsAllZeros.hpp>
#include <probes/ProbeData.hpp>
#include <probes/StatsProbeTypes.hpp>
#include <utils/PVLog.hpp>

#include <cstdlib>
#include <ctime>
#include <map>
#include <string>
#include <vector>

using PV::CheckStatsAllZeros;
using PV::LayerStats;
using PV::ProbeData;
using PV::PV_Init;
using PV::PVParams;

void initRNG();
ProbeData<LayerStats> initStatsBatch(double timestamp, std::vector<int> numNonzeroVector);
CheckStatsAllZeros initTestObject(PV_Init const &pv_init, bool exitFlag, bool immediateExitFlag);
int run(PV_Init const &pv_init);
int run(PV_Init const &pv_init, std::vector<int> const &numNonzeroVector);

int main(int argc, char **argv) {
   PV_Init pv_init(&argc, &argv, false);
   initRNG();
   int status = run(pv_init);
   if (status == PV_SUCCESS) {
      InfoLog() << "Test passed.\n";
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

void initRNG() {
   unsigned int seed = std::time((std::time_t *)nullptr);
   PV::RandomSeed::instance()->initialize(seed);
}

ProbeData<LayerStats> initStatsBatch(double timestamp, std::vector<int> numNonzeroVector) {
   auto rng = PV::Random(1);
   ProbeData<LayerStats> statsBatch(timestamp, numNonzeroVector.size());
   int batchWidth = static_cast<int>(numNonzeroVector.size());
   for (int b = 0; b < batchWidth; ++b) {
      PV::LayerStats &stats = statsBatch.getValue(b);
      stats.mSum            = static_cast<double>(rng.uniformRandom());
      stats.mSumSquared     = static_cast<double>(rng.uniformRandom());
      stats.mMin            = rng.uniformRandom(0, -2.0f, 0.0f);
      stats.mMax            = rng.uniformRandom(0, 0.0f, 2.0f);
      stats.mNumNeurons     = 64;
      stats.mNumNonzero     = numNonzeroVector[b]; // number outside of threshold
   }
   return statsBatch;
}

CheckStatsAllZeros initTestObject(PV_Init const &pv_init, bool exitFlag, bool immediateExitFlag) {
   std::string exitString(exitFlag ? "true" : "false");
   std::string immediateExitString(immediateExitFlag ? "true" : "false");
   std::string paramsString;
   paramsString.append("debugParsing = false;\n");
   paramsString.append("CheckStatsAllZeros \"TestObject\" = {\n");
   paramsString.append("   exitOnFailure = ").append(exitString).append(";\n");
   paramsString.append("   immediateExitOnFailure = ").append(immediateExitString).append(";\n");
   paramsString.append("};\n");

   MPI_Comm mpiComm = pv_init.getCommunicator()->globalCommunicator();
   PVParams params(paramsString.data(), paramsString.size(), 1UL, mpiComm);

   CheckStatsAllZeros testObject("TestObject", &params);
   testObject.ioParamsFillGroup(PV::PARAMS_IO_READ);
   return testObject;
}

int run(PV_Init const &pv_init) {
   int status = PV_SUCCESS;
   std::string infoMessage;

   InfoLog().printf("\nChecking CheckStatsAllZeros::checkStats() with nonzero values present:\n");
   std::vector<int> numNonzeroVector{0, 0, 1, 0, 2, 3};
   if (run(pv_init, numNonzeroVector) != PV_SUCCESS) {
      status = PV_FAILURE;
   }

   InfoLog().printf("\nChecking CheckStatsAllZeros::checkStats() with all zero values:\n");
   std::vector<int> allZeros{0, 0, 0, 0, 0, 0};
   if (run(pv_init, allZeros) != PV_SUCCESS) {
      status = PV_FAILURE;
   }

   return status;
}

int run(PV_Init const &pv_init, std::vector<int> const &numNonzeroVector) {
   std::vector<int> nonzeroElements;
   int numAllElements = static_cast<int>(numNonzeroVector.size());
   std::string infoMessage;
   for (int n = 0; n < numAllElements; ++n) {
      if (numNonzeroVector[n]) {
         if (nonzeroElements.empty()) {
            infoMessage.append("There should be an error message listing element(s) ");
         }
         else {
            infoMessage.append(", ");
         }
         infoMessage.append(std::to_string(n));
         nonzeroElements.push_back(n);
      }
   }
   if (nonzeroElements.empty()) {
      infoMessage.append("This checkStats() call should not generate any error message.\n");
   }
   else {
      infoMessage.append(".\n");
   }
   InfoLog() << infoMessage;
   bool exitOnFailure            = false;
   bool immediateExitOnFailure   = false;
   CheckStatsAllZeros testObject = initTestObject(pv_init, exitOnFailure, immediateExitOnFailure);
   ProbeData<LayerStats> statsBatch                = initStatsBatch(1.0, numNonzeroVector);
   std::map<int, PV::LayerStats const> checkResult = testObject.checkStats(statsBatch);

   FatalIf(
         statsBatch.size() != numNonzeroVector.size(),
         "Size of test ProbeData object is %zu instead of expected %zu\n",
         std::size_t(statsBatch.size()),
         std::size_t(numNonzeroVector.size()));

   int batchSize = static_cast<int>(numNonzeroVector.size());
   for (int b = 0; b < batchSize; ++b) {
      auto const found = checkResult.find(b);
      if (numNonzeroVector[b]) {
         FatalIf(
               found == checkResult.end(),
               "CheckStatsAllZeros::checkStats() did not flag stats element %d "
               "even though its NumNonzero value is %d\n",
               b,
               statsBatch.getValue(b).mNumNonzero);
      }
      else {
         FatalIf(
               found != checkResult.end(),
               "CheckStatsAllZeros::checkStats() flagged stats element %d "
               "as having NumNonzero value of %d, even though it should be 0.\n",
               b,
               found->second.mNumNonzero);
      }
   }

   return PV_SUCCESS;
}
