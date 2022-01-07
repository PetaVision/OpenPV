#include "PathComponents.hpp"
#include <cerrno>
#include <cstring>
#include <libgen.h>
#include <stdexcept>

namespace PV {

std::string dirName(std::string const &path) {
   return dirName(path.c_str());  
}

std::string dirName(char const *path) {
   char *pathCopy = strdup(path);
   if (pathCopy == nullptr) {
      std::string errorMessage("dirName failed to allocate memory: ");
      errorMessage.append(strerror(errno));
      throw std::runtime_error(errorMessage);
   }
   char *directoryPart = dirname(pathCopy);
   std::string result(directoryPart);
   free(pathCopy);
   return result;
}

std::string baseName(std::string const &path) {
   return baseName(path.c_str());  
}

std::string baseName(char const *path) {
   char *pathCopy = strdup(path);
   if (pathCopy == nullptr) {
      std::string errorMessage("baseName failed to allocate memory: ");
      errorMessage.append(strerror(errno));
      throw std::runtime_error(errorMessage);
   }
   char *basenamePart = basename(pathCopy);
   std::string result(basenamePart);
   free(pathCopy);
   return result;
}

std::string extension(std::string const &path) {
   auto basename = baseName(path);
   auto lastdot = basename.rfind('.');
   std::string ext;
   if (lastdot != std::string::npos) {
      ext = basename.substr(lastdot, std::string::npos);
   }
   return ext;
}

std::string extension(char const *path) {
   std::string pathString(path);
   return extension(pathString);
}

std::string stripExtension(std::string const &path) {
   auto basename = baseName(path);
   auto lastdot = basename.rfind('.');
   std::string ext = basename.substr(0, lastdot);
   return ext;
}

std::string stripExtension(char const *path) {
   std::string pathString(path);
   return stripExtension(pathString);
}

} // namespace PV
