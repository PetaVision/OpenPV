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

   FatalIf(configParser.getBooleanArgument("RequireReturn")!=true, "Parsing RequireReturn failed.\n");
   FatalIf(configParser.getBooleanArgument("Restart")!=false, "Parsing Restart failed.\n");
   FatalIf(configParser.getBooleanArgument("DryRun")!=true, "Parsing DryRun failed.\n");
   FatalIf(configParser.getUnsignedIntArgument("RandomSeed")!=1234565432U, "Parsing RandomSeed failed.\n");
   PV::Configuration::IntOptional numThreads = configParser.getIntOptionalArgument("NumThreads");
   FatalIf(numThreads.mUseDefault!=false, "Parsing NumThreads failed.\n");
   FatalIf(numThreads.mValue!=8, "Parsing NumThreads failed.\n");
   FatalIf(configParser.getIntegerArgument("NumRows")!=2, "Parsing NumRows failed.\n");
   FatalIf(configParser.getIntegerArgument("NumColumns")!=3, "Parsing NumColumns failed.\n");
   FatalIf(configParser.getIntegerArgument("BatchWidth")!=4, "Parsing BatchWidth failed.\n");
   FatalIf(configParser.getStringArgument("OutputPath")!= "outputPath", "Parsing OutputPath failed.\n");
   FatalIf(configParser.getStringArgument("ParamsFile")!= "input/pv.params", "Parsing ParamsFile failed.\n");
   FatalIf(configParser.getStringArgument("LogFile")!= "test.log", "Parsing LogFile failed.\n");
   FatalIf(configParser.getStringArgument("GPUDevices")!= "0,1", "Parsing GpuDevices failed.\n");
   FatalIf(configParser.getStringArgument("WorkingDirectory")!= ".", "Parsing WorkingDir failed.\n");
   FatalIf(configParser.getStringArgument("CheckpointReadDirectory")!= "outputPath/checkpoints", "Parsing CheckpointReadDirectory failed.\n");
   return 0;
}
