/*
 * pv.cpp
 */

#include <string>

#include <columns/buildandrun.hpp>
#include <utils/PathComponents.hpp> // baseName()

int main(int argc, char *argv[]) {
   PV_Init pv_init(&argc, &argv, false /*allowUnrecognizedArgumentsFlag*/);
   int status = buildandrun(&pv_init);
   FatalIf(status != PV_SUCCESS, "%s failed.\n", baseName(argv[0]).c_str());

   auto fileManager = pv_init.getCommunicator()->getOutputFileManager();
   if (!fileManager->isRoot()) {
      return EXIT_SUCCESS;
   }

   auto baseInputIndicesFile = fileManager->open("timestamps/BaseInput.txt", std::ios_base::in);
   long baseFileLength = baseInputIndicesFile->getFileSize();
   auto syncedInputIndicesFile = fileManager->open("timestamps/SyncedInput.txt", std::ios_base::in);
   long syncedFileLength = syncedInputIndicesFile->getFileSize();

   std::string keyString = "index:";
   while (baseInputIndicesFile->getInPos() < baseFileLength) {
      std::string baseInputLine = baseInputIndicesFile->readLine();
      std::string syncedInputLine = syncedInputIndicesFile->readLine();
      auto basePos = baseInputLine.find(keyString);
      auto syncedPos = syncedInputLine.find(keyString);
      if (basePos == std::string::npos or syncedPos == std::string::npos) {
         ErrorLog().printf("Unexpected format of timestamps file.\n");
         status = PV_FAILURE;
         break;
      }
      std::string baseIndexString = baseInputLine.substr(basePos + keyString.size());
      int baseIndex = std::stoi(baseIndexString);
      std::string syncedIndexString = syncedInputLine.substr(syncedPos + keyString.size());
      int syncedIndex = std::stoi(syncedIndexString);
      if (syncedIndex != baseIndex) {
         ErrorLog().printf(
               "SyncedInput index did not match BaseInput index: \n"
               "    BaseInput line:   \"%s\"\n"
               "    SyncedInput line: \"%s\"\n",
               baseInputLine.c_str(), syncedInputLine.c_str());
         status = PV_FAILURE;
      }
   }
   FatalIf(status != PV_SUCCESS, "%s failed.\n", baseName(argv[0]).c_str());
   return EXIT_SUCCESS;
}
