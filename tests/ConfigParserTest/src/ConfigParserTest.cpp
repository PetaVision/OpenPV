/*
 * ConfigParserTest.cpp
 *
 */

#include <io/ConfigParser.hpp>
#include <utils/PVLog.hpp>
#include <fstream>
#include <string>

int main(int argc, char *argv[]) {
   std::string programName{argv[0]};
   std::size_t lastSlash = programName.rfind('/');
   std::string logFile;
   if (lastSlash == std::string::npos) {
      logFile = programName;
   }
   else {
      logFile = programName.substr(lastSlash+1, std::string::npos);
   }
   logFile.append("_1.log");
   PV::setLogFile(logFile);

   bool configFromArgv = argc > 1 && argv != nullptr && argv[1] != nullptr && argv[1][0] != '\0';
   char const *configFile = configFromArgv ? argv[1] : "input/config.txt";
   std::ifstream inputStream{configFile};
   FatalIf(inputStream.fail(), "failed to open %s\n", configFile);
   PV::ConfigParser configParser{inputStream, false /*do not allow unrecognized arguments*/};

   FatalIf(configParser.getRequireReturn()!=true, "Parsing RequireReturn failed.\n");
   FatalIf(configParser.getRestart()!=false, "Parsing Restart failed.\n");
   FatalIf(configParser.getDryRun()!=true, "Parsing DryRun failed.\n");
   FatalIf(configParser.getRandomSeed()!=1234565432U, "Parsing RandomSeed failed.\n");
   FatalIf(configParser.getUseDefaultNumThreads()!=false, "Parsing NumThreads failed.\n");
   FatalIf(configParser.getNumThreads()!=8, "Parsing NumThreads failed.\n");
   FatalIf(configParser.getNumRows()!=2, "Parsing NumRows failed.\n");
   FatalIf(configParser.getNumColumns()!=3, "Parsing NumColumns failed.\n");
   FatalIf(configParser.getBatchWidth()!=4, "Parsing BatchWidth failed.\n");
   FatalIf(configParser.getOutputPath()!= "outputPath", "Parsing OutputPath failed.\n");
   FatalIf(configParser.getParamsFile()!= "input/pv.params", "Parsing ParamsFile failed.\n");
   FatalIf(configParser.getLogFile()!= "test.log", "Parsing LogFile failed.\n");
   FatalIf(configParser.getGpuDevices()!= "0,1", "Parsing GpuDevices failed.\n");
   FatalIf(configParser.getWorkingDir()!= ".", "Parsing WorkingDir failed.\n");
   FatalIf(configParser.getCheckpointReadDir()!= "outputPath/checkpoints", "Parsing CheckpointReadDirectory failed.\n");
   return 0;
}
