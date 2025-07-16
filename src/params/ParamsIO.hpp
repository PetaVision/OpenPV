#ifndef PARAMSIO_HPP_
#define PARAMSIO_HPP_

#include "include/pv_common.h"
#include "io/FileStream.hpp"
#include "params/ParamGroup.hpp"
#include <cassert>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace PV {

enum class ParamsIOSwitch { Read, Write };

class ParamsIO {
  public:

   ParamsIO(std::shared_ptr<ParamGroup> params, std::shared_ptr<ParamGroup> defaults = nullptr);

   Parameter::Type checkType(std::string const &paramName) const;

   void handleUnnecessaryParameter(std::string const &paramName);

   template <typename T>
   typename std::enable_if<std::is_arithmetic<T>::value, void>::type
   handleUnnecessaryParameter(std::string const &paramName, T correctValue);

   template <typename T>
   typename std::enable_if<std::is_same<T, std::string>::value, void>::type
   handleUnnecessaryParameter(std::string const &paramName, T correctValue);

   template <typename T>
   typename std::enable_if<
         std::is_same<
               T,
               typename std::vector<typename T::value_type, typename T::allocator_type>>::value,
         void>::type
   handleUnnecessaryParameter(std::string const &paramName, T correctValue);

   void handleUnnecessaryCaseInsensitiveParameter(
      std::string const &param_name, std::string const &correct_value);

   bool hasBeenRead(char const *param_name);

   template <typename T>
   void ioParam(
         ParamsIOSwitch ioSwitch,
         std::string const &paramName,
         T *value,
         bool warnIfAbsentFlag = true);

   bool isPresent(std::string const &paramName);

   bool isArray(std::string const &paramName) const;
   bool isNumeric(std::string const &paramName) const;
   bool isString(std::string const &paramName) const;

   bool presentAndNotBeenRead(const char *param_name);

   template <typename T>
   typename std::enable_if<std::is_arithmetic<T>::value, T>::type
   readValue(std::string const &paramName, bool warnIfAbsentFlag = true);

   template <typename T>
   typename std::enable_if<std::is_same<T, std::string>::value, T>::type
   readValue(std::string const &paramName, bool warnIfAbsentFlag = true);

   template <typename T>
   typename std::enable_if<
         std::is_same<
               T,
               typename std::vector<typename T::value_type, typename T::allocator_type>>::value,
         T>::type
   readValue(std::string const &paramName, bool warnIfAbsentFlag = true);

   std::string const &getName() const { return mParams->getName(); }
   std::string const &getKeyword() const { return mParams->getKeyword(); }

   std::shared_ptr<ParamGroup> getParams() { return mParams; }
   std::shared_ptr<ParamGroup> getDefaults() { return mDefaults; }

   FileStream *getPrintParamsStream() { return mPrintParamsStream; }
   FileStream *getPrintLuaStream() { return mPrintLuaStream; }

   void setPrintParamsStream(FileStream *stream) { mPrintParamsStream = stream; }
   void setPrintLuaStream(FileStream *stream) { mPrintLuaStream = stream; }

  private:
   template <typename T>
   T convertDoubleToArithmeticType(double value);

   std::vector<double> const *readArray(std::string const &paramName, bool warnIfAbsentFlag);
   double readDouble(std::string const &paramName, bool warnIfAbsentFlag);
   std::string const &readString(std::string const &paramName, bool warnIfAbsentFlag);

   template <typename T>
   void writeParam(std::string const &paramName, T const &paramValue);

   template <typename T>
   typename std::enable_if<std::is_arithmetic<T>::value, std::string>::type
   paramToString(T const &paramValue);

   template <typename T>
   typename std::enable_if<std::is_same<T, std::string>::value, std::string>::type
   paramToString(T const &paramValue);

   template <typename T>
   typename std::enable_if<
         std::is_same<
               T,
               typename std::vector<typename T::value_type, typename T::allocator_type>>::value,
         std::string>::type
   paramToString(T const &paramValue);
  
  private:
   std::shared_ptr<ParamGroup> mParams;
   std::shared_ptr<ParamGroup> mDefaults;

   FileStream *mPrintParamsStream = nullptr;
   FileStream *mPrintLuaStream    = nullptr;
};

template <typename T>
T ParamsIO::convertDoubleToArithmeticType(double value) {
   return static_cast<T>(value);
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, void>::type
ParamsIO::handleUnnecessaryParameter(std::string const &paramName, T correctValue) {
   handleUnnecessaryParameter(paramName);
   double const *valueAsDouble = mParams->read<double>(paramName);
   if (valueAsDouble == nullptr) { return; }
   T value = convertDoubleToArithmeticType<T>(*valueAsDouble);
   if (value != correctValue) {
      Fatal() << "Value " << value << " is inconsistent with correct value "
              << correctValue << " (discrepancy " << value - correctValue << ")\n";
   }
}

template <typename T>
typename std::enable_if<std::is_same<T, std::string>::value, void>::type
ParamsIO::handleUnnecessaryParameter(std::string const &paramName, T correctValue) {
   handleUnnecessaryParameter(paramName);
   std::string const *value = mParams->read<std::string>(paramName);
   if (value == nullptr) { return; }
   if (*value != correctValue) {
      Fatal() << "Value \"" << value << "\" is inconsistent with correct value \""
              << correctValue << "\"\n";
   }
}

template <typename T>
typename std::enable_if<
      std::is_same<
            T,
            typename std::vector<typename T::value_type, typename T::allocator_type>>::value,
      void>::type
ParamsIO::handleUnnecessaryParameter(std::string const &paramName, T correctValue) {
   handleUnnecessaryParameter(paramName);
   std::vector<double> const *value = mParams->read<T>(paramName);
   if (value == nullptr) { return; }
   if (value->size() != correctValue.size()) {
      Fatal() << "Value has size " << value->size() << " but correct value has size "
              << correctValue.size() << "\n";
   }
   int status = PV_SUCCESS;
   std::size_t N = correctValue.size();
   assert(value->size() == N);
   for (std::size_t n = 0; n < N; ++n) {
      typename T::value_type correct  = correctValue[n];
      typename T::value_type observed = static_cast<typename T::value_type>(value->at(n));
      if (observed != correct) {
         ErrorLog() << "Element " << n + 1UL << "value " << observed
                    << " is inconsistent with correct value " << correct
                    << " (discrepancy " << observed - correct << ")\n";
         status = PV_FAILURE;
      }
      FatalIf(
            status != PV_SUCCESS,
            "%s \"%s\" array parameter %s has incorrect value.\n", 
            getKeyword().c_str(), getName().c_str(), paramName);
   }
}

template <typename T>
void ParamsIO::ioParam(
      ParamsIOSwitch ioSwitch,
      std::string const &paramName,
      T *value,
      bool warnIfAbsentFlag) {
   switch(ioSwitch) {
      case ParamsIOSwitch::Read:
         *value = readValue<T>(paramName, warnIfAbsentFlag);
         break;
      case ParamsIOSwitch::Write:
         writeParam<T>(paramName, *value);
         break;
      default:
         assert(0); // All possibilities for ioSwitch are handled above
         break;
   }
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
ParamsIO::readValue(std::string const &paramName, bool warnIfAbsentFlag) {
   double valueAsDouble = readDouble(paramName, warnIfAbsentFlag);
   return convertDoubleToArithmeticType<T>(valueAsDouble);
}

template <typename T>
typename std::enable_if<std::is_same<T, std::string>::value, T>::type
ParamsIO::readValue(std::string const &paramName, bool warnIfAbsentFlag) {
   std::string const &string = readString(paramName, warnIfAbsentFlag);
   return string;
}

template <typename T>
typename std::enable_if<
      std::is_same<
            T,
            typename std::vector<typename T::value_type, typename T::allocator_type>>::value,
      T>::type
ParamsIO::readValue(std::string const &paramName, bool warnIfAbsentFlag) {
   std::vector<double> const *values = readArray(paramName, warnIfAbsentFlag);
   auto N = values->size();
   std::vector<typename T::value_type> result(values->size());
   for (decltype(N) n = 0; n < N; ++n) {
      result[n] = static_cast<typename T::value_type>(values->at(n));
   }
   return result;
}

template <typename T>
void ParamsIO::writeParam(std::string const &paramName, T const &paramValue) {
   if (mPrintParamsStream) {
      mPrintParamsStream->printf(
            "    %-35s = %s;\n", paramName.c_str(), paramToString<T>(paramValue).c_str());
   }
   if (mPrintLuaStream) {
      mPrintLuaStream->printf(
            "    %-35s = %s;\n", paramName.c_str(), paramToString<T>(paramValue).c_str());
   }
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value, std::string>::type
ParamsIO::paramToString(T const &paramValue) {
   std::string result;
   if (std::numeric_limits<T>::has_infinity) {
      if (paramValue == std::numeric_limits<T>::lowest()) {
         result = "-infinity";
      }
      else if (paramValue == std::numeric_limits<T>::max()) {
         result = "infinity";
      }
      else {
         result = std::to_string(paramValue);
      }
   }
   else {
      result = std::to_string(paramValue);
   }
   return result;
}

template <typename T>
typename std::enable_if<std::is_same<T, std::string>::value, std::string>::type
ParamsIO::paramToString(T const &paramValue) {
   if (paramValue.empty()) {
      return std::string("NULL");
   }
   else {
      return "\"" + paramValue + "\"";
   }
}

template <typename T>
typename std::enable_if<
      std::is_same<
            T,
            typename std::vector<typename T::value_type, typename T::allocator_type>>::value,
      std::string>::type
ParamsIO::paramToString(T const &paramValue) {
   std::string result = "[";
   if (!paramValue.empty()) {
      int arraysize = paramValue.size();
      for (int k = 0; k < arraysize - 1; ++k) {
         result.append(paramToString(paramValue[k])).append(",");
      }
      result.append(paramToString(paramValue[arraysize - 1]));
   }
   result += ']';
   return result;
}

} // end namespace PV

#endif // PARAMSIO_HPP_
