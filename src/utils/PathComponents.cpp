#include "PathComponents.hpp"
#include <cassert>
#include <cerrno>
#include <cstring>
#include <libgen.h>
#include <stdexcept>

namespace PV {

std::string dirName(std::string const &path) {
   std::string pathCopy = path;
   assert(pathCopy.data() != path.data());
   char *directoryPart = dirname(&pathCopy.at(0));
   std::string result(directoryPart);
   return result;
}

std::string dirName(char const *path) {
   std::string pathCopy(path);
   char *directoryPart = dirname(&pathCopy.at(0));
   std::string result(directoryPart);
   return result;
}

std::string baseName(std::string const &path) {
   std::string pathCopy(path);
   assert(pathCopy.data() != path.data());
   char *basenamePart = basename(&pathCopy.at(0));
   std::string result(basenamePart);
   return result;
}

std::string baseName(char const *path) {
   std::string pathCopy(path);
   char *basenamePart = basename(&pathCopy.at(0));
   std::string result(basenamePart);
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

std::string stripExtension(std::string const &path) {
   auto basename = baseName(path);
   auto lastdot = basename.rfind('.');
   std::string ext = basename.substr(0, lastdot);
   return ext;
}

} // namespace PV
