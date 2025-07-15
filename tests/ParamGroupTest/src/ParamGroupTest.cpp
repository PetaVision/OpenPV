/**
 * ParamGroupTest.cpp
 */

#include <include/pv_common.h>
#include <io/io.hpp>
#include <params/Parameter.hpp>
#include <params/ParamGroup.hpp>
#include <utils/PVLog.hpp>

using namespace PV;

int main(int argc, char **argv) {
   auto arguments = parse_arguments(argc, argv, false /*allowUnrecognizedArgumentsFlag*/);
   std::string logfile = arguments->getStringArgument("LogFile");
   setLogFile(logfile);

   int status = PV_SUCCESS;

   ParamGroup testParamGroup("Name", "Keyword", 0 /*processRank*/);
   std::string const &checkName = testParamGroup.getName();
   FatalIf(checkName != "Name", "ParameterGroup Name property failed.\n");
   std::string const &checkKeyword = testParamGroup.getKeyword();
   FatalIf(checkKeyword != "Keyword", "ParameterGroup Keyword property failed.\n");

   testParamGroup.insert("float", 3.5);
   testParamGroup.insert("integer", -8);
   testParamGroup.insert("bool", true);
   std::vector<double> testArray{1.25, 2.5, 5.0, 10.0};
   testParamGroup.insert("array", testArray);
   testParamGroup.insert("string", "A string");

   double const *intValuePtr = testParamGroup.read<double>("integer");
   FatalIf(!intValuePtr, "Failed to read parameter \"integer\"\n");
   FatalIf(
         *intValuePtr != -8.0,
         "Reading parameter \"integer\" gave the value %d instead of expected -8\n",
         *intValuePtr);

   std::string const *stringValuePtr = testParamGroup.read<std::string>("string");
   FatalIf(!stringValuePtr, "Failed to read parameter \"string\"\n");
   FatalIf(
         *stringValuePtr != "A string",
         "Reading parameter \"string\" gave the value \"%s\" instead of expected \"A string\"\n",
         stringValuePtr->c_str());

   double const *boolValuePtr = testParamGroup.read<double>("bool");
   FatalIf(!boolValuePtr, "Failed to read parameter \"bool\"\n");
   FatalIf(
         *boolValuePtr != true,
         "Reading parameter \"bool\" gave the value false instead of expected true\n");

   std::vector<double> const *arrayValuePtr = testParamGroup.read<std::vector<double>>("array");
   FatalIf(!arrayValuePtr, "Failed to read parameter \"array\"\n");
   FatalIf(
         *arrayValuePtr != testArray,
         "Reading parameter \"array\" did not give expected array values.\n");

   // We haven't read "float" yet. Let's see if lookForUnread() works.
   auto unreadParams = testParamGroup.lookForUnread();
   FatalIf(
         unreadParams.empty(),
         "lookForUnread() was empty when there should have been an unread parameter.\n");
   // Now read it and check its value
   auto floatValuePtr = testParamGroup.read<double>("float");
   FatalIf(!floatValuePtr, "Failed to read parameter \"float\"\n");
   FatalIf(
         *floatValuePtr != 3.5,
         "Reading parameter \"float\" gave the value %f instead of expected 3.5\n",
         static_cast<double>(*floatValuePtr));
   // Then see if lookForUnread() reports that all parameters have been read.
   unreadParams = testParamGroup.lookForUnread();
   FatalIf(
         !unreadParams.empty(),
         "lookForUnread() was non-empty when every parameter should have been read.\n");

   // Clear all the HasBeenRead flags, and test lookForUnread() again
   testParamGroup.clearAllHasBeenReadFlags();
   unreadParams = testParamGroup.lookForUnread();
   FatalIf(
         unreadParams.empty(),
         "lookForUnread() was empty when all parameters should have been unread.\n");

   // Erase a parameter, checking before and after that present() returns the correct value
   bool boolIsPresent = testParamGroup.present("bool");
   FatalIf(!boolIsPresent, "present(\"bool\") returned false when it should be true.\n");
   bool erased = testParamGroup.erase("bool");
   FatalIf(!erased, "erase(\"bool\") returned false when it should be true.\n");
   boolIsPresent = testParamGroup.present("bool");
   FatalIf(boolIsPresent, "present(\"bool\") returned true when it should be false.\n");

   bool findNonexistent = testParamGroup.present("apple");
   FatalIf(findNonexistent, "present(\"apple\") returned true when it should be false.\n");

   auto nonexistentArrayPtr = testParamGroup.read<std::vector<double>>("apple");
   FatalIf(
         nonexistentArrayPtr,
         "reading nonexistent parameter as array returned true when it should be false.\n");

   auto nonexistentFloatPair = testParamGroup.read<double>("banana");
   FatalIf(
         nonexistentFloatPair,
         "reading nonexistent parameter as numeric returned true when it should be false.\n");

   stringValuePtr = testParamGroup.read<std::string>("banana");
   FatalIf(
         stringValuePtr,
         "reading nonexistent parameter as string returned true when it should be false.\n");

   stringValuePtr = testParamGroup.read<std::string>("float");
   FatalIf(
         stringValuePtr,
         "reading numeric parameter as string returned true when it should be false.\n");

   testParamGroup.insert("String", "");
   Parameter::Type type = testParamGroup.checkType("String");
   FatalIf(
         type != Parameter::Type::String,
         "checkType(\"String\") returned %d when it should be %d (STRING).\n",
         type, Parameter::Type::String);
   stringValuePtr = testParamGroup.read<std::string>("String");
   FatalIf(!stringValuePtr, "Failed to read parameter \"String\"\n");

   bool insertExisting = testParamGroup.insert("integer", 3);
   FatalIf(
         insertExisting,
         "insert() with existing parameter name returned true when it should be false.\n");
   insertExisting = testParamGroup.insert("integer", "three");
   FatalIf(
         insertExisting,
         "insert() with existing parameter name returned true when it should be false.\n");

   return status;
}
