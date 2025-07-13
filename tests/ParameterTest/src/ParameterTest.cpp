/**
 * ParameterTest.cpp
 */

#include <include/pv_common.h>
#include <io/io.hpp>
#include <params/Parameter.hpp>
#include <utils/PVLog.hpp>

#include <cstring> // std::strcmp()
#include <string>

using namespace PV;

using PVArray = std::vector<double>;

template<typename ReadType>
typename std::enable_if<std::is_same<ReadType, double>::value, bool>::type
checkValue(ReadType readValue, ReadType correct);

template <typename ReadType>
typename std::enable_if<
      std::is_same<
            ReadType,
            std::vector<typename ReadType::value_type, typename ReadType::allocator_type>>::value,
      bool>::type
checkValue(ReadType readValue, ReadType correct);

template<typename CreateType, typename ReadType, bool canRead>
int testParam(std::string const &name, CreateType value, ReadType correct);

int main(int argc, char **argv) {
   auto arguments = parse_arguments(argc, argv, false /*allowUnrecognizedArgumentsFlag*/);
   std::string logfile = arguments->getStringArgument("LogFile");
   setLogFile(logfile);

   int status = PV_SUCCESS;
   int testResult;

   // Create a numeric parameter (type double) and see if it reads correctly
   testResult = testParam<double, double, true>("doubleParam", 3.5, 3.5);
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   testResult = testParam<double, PVArray, true>("readDoubleAsVector", 3.5, PVArray{3.5});
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   testResult = testParam<double, std::string, false>("readDoubleAsString", 3.5, "");
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   // Create an array parameter of length 1 and see if it reads correctly
   testResult = testParam<PVArray, double, true>("arrayLength1asDouble", PVArray{3.5}, 3.5);
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   testResult = testParam<PVArray, PVArray, true>("arrayLength1", PVArray{3.5}, PVArray{3.5});
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   testResult = testParam<PVArray, std::string, false>("arrayLength1asString", PVArray{3.5}, "");
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   // Create an array parameter of length greater than 1 and see if it reads correctly
   PVArray pvArray{2.5, 4.0, 6.5};

   testResult = testParam<PVArray, double, false>("arrayLength3asDouble", pvArray, 0.0);
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   testResult = testParam<PVArray, PVArray, true>("arrayLength3asArray", pvArray, pvArray);
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   testResult = testParam<PVArray, std::string, false>("arrayLength3asString", pvArray, "");
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   // Create a string parameter and see if it reads correctly
   char const *testString = "PetaVision";

   testResult = testParam<std::string, double, false>("stringAsDouble", testString, 0.0);
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   testResult = testParam<std::string, PVArray, false>("stringAsArray", testString, pvArray);
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   testResult = testParam<std::string, std::string, true>("stringAsString", testString, testString);
   if (testResult != PV_SUCCESS) { status = PV_FAILURE; }

   if (status == PV_SUCCESS) {
      InfoLog().printf("Test passed.\n");
   }
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

template<typename ReadType>
typename std::enable_if<std::is_same<ReadType, double>::value, bool>::type
checkValue(ReadType readValue, ReadType correct) {
   double discrepancy = readValue - correct;
   if (discrepancy) {
      ErrorLog().printf(
            "Expected %f, received %f, discrepancy %g\n",
            static_cast<double>(readValue),
            static_cast<double>(correct),
            static_cast<double>(discrepancy));
   }
   return discrepancy == 0.0;
}

template <typename ReadType>
typename std::enable_if<
      std::is_same<
            ReadType,
            std::vector<typename ReadType::value_type, typename ReadType::allocator_type>>::value,
      bool>::type
checkValue(ReadType readValue, ReadType correct) {
   auto N = readValue.size();
   if (correct.size() != N) {
      ErrorLog().printf(
            "Expected a vector of size %zu; received a vector of size %zu\n",
            correct.size(),
            N);
      return false;
   }
   else {
      bool isCorrect = true;
      for (decltype(N) n = 0UL; n < N; ++n) {
         if (readValue[n] != correct[n]) {
             ErrorLog().printf(
                   "Element %zu: expected %f, received %f, discrepancy %g\n",
                   n,
                   static_cast<double>(correct[n]),
                   static_cast<double>(readValue[n]),
                   static_cast<double>(readValue[n] - correct[n]));
             isCorrect = false;
         }
      }
      return isCorrect;
   }
}

template<typename ReadType>
typename std::enable_if<std::is_same<ReadType, std::string>::value, bool>::type
checkValue(std::string readValue, std::string correct) {
   bool isCorrect = readValue == correct;
   if (!isCorrect) {
      ErrorLog().printf(
            "Expected \"%s\", received \"%s\"\n", correct.c_str(), readValue.c_str());
   }
   return isCorrect;
}

template<typename CreateType, typename ReadType, bool canRead>
int testParam(std::string const &name, CreateType value, ReadType correct) {
   int status = PV_SUCCESS;
   Parameter p(value);

   if (p.getHasBeenReadFlag()) {
      ErrorLog().printf("%s HasBeenReadFlag is true when it should be false\n", name.c_str());
      status = PV_FAILURE;
   }
   ReadType const *readValuePtr = p.read<ReadType>();
   bool hasValue = readValuePtr != nullptr;
   if (canRead and !hasValue) {
      ErrorLog().printf("%s read() failed when it should have succeeded.\n", name.c_str());
      status = PV_FAILURE;
   }
   if (!canRead and hasValue) {
      ErrorLog().printf("%s read() succeeded when it should have failed.\n", name.c_str());
      status = PV_FAILURE;
   }

   if (hasValue) {
      if (!p.getHasBeenReadFlag()) {
         ErrorLog().printf("%s HasBeenReadFlag is false when it should be true\n", name.c_str());
         status = PV_FAILURE;
      }
      ReadType const readValue = *readValuePtr;
      bool isCorrect = checkValue<ReadType>(readValue, correct);
      if (!isCorrect) {
         ErrorLog().printf("%s read() function returned an incorrect value\n", name.c_str());
         status = PV_FAILURE;
      }
   }
   else {
      if (p.getHasBeenReadFlag()) {
         ErrorLog().printf("%s HasBeenReadFlag is true when it should be false\n", name.c_str());
         status = PV_FAILURE;
      }
   }

   p.clearHasBeenReadFlag();
   if (p.getHasBeenReadFlag()) {
      ErrorLog().printf("%s HasBeenReadFlag is true when it should be false\n", name.c_str());
      status = PV_FAILURE;
   }

   return status;
}
