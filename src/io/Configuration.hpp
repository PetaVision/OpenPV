#ifndef CONFIGURATION_HPP_
#define CONFIGURATION_HPP_

#include <map>
#include <string>
#include <vector>

namespace PV {

class Configuration {
  public:
   struct IntOptional {
      bool mUseDefault = false;
      int mValue       = -1;
   };
   enum ConfigurationType {
      CONFIG_UNRECOGNIZED,
      CONFIG_BOOL,
      CONFIG_INT,
      CONFIG_UNSIGNED,
      CONFIG_STRING,
      CONFIG_INT_OPTIONAL
   };

   Configuration();
   ~Configuration() {}

   /**
    * Returns the type for the given string:
    * unrecognized, boolean, integer, unsigned, string, or optional integer.
    */
   ConfigurationType getType(std::string const &name) const;

   bool const &getBooleanArgument(std::string const &name) const;
   int const &getIntegerArgument(std::string const &name) const;
   unsigned int const &getUnsignedIntArgument(std::string const &name) const;
   std::string const &getStringArgument(std::string const &name) const;
   IntOptional const &getIntOptionalArgument(std::string const &name) const;

   static std::string printBooleanArgument(std::string const &name, bool const &value);

   static std::string printIntegerArgument(std::string const &name, int const &value);

   static std::string printUnsignedArgument(std::string const &name, unsigned int const &value);

   static std::string printStringArgument(std::string const &name, std::string const &value);

   static std::string printIntOptionalArgument(std::string const &name, IntOptional const &value);

   std::string printArgument(std::string const &name) const;

   std::string printConfig() const;

   bool setArgumentUsingString(std::string const &name, std::string const &value);
   bool setBooleanArgument(std::string const &name, bool const &value);
   bool setIntegerArgument(std::string const &name, int const &value);
   bool setUnsignedIntArgument(std::string const &name, unsigned int const &value);
   bool setStringArgument(std::string const &name, std::string const &value);
   bool setIntOptionalArgument(std::string const &name, IntOptional const &value);

  private:
   void registerArgument(std::string const &name, ConfigurationType type);
   void registerBooleanArgument(std::string const &name);
   void registerIntegerArgument(std::string const &name);
   void registerUnsignedIntArgument(std::string const &name);
   void registerStringArgument(std::string const &name);
   void registerIntOptionalArgument(std::string const &name);

   bool parseBoolean(std::string const &valueString) const;
   int parseInteger(std::string const &valueString) const;
   unsigned int parseUnsignedInt(std::string const &valueString) const;
   std::string parseString(std::string const &valueString) const;
   IntOptional parseIntOptional(std::string const &valueString) const;

   // Data members
  private:
   std::vector<std::string> mConfigArguments;
   std::map<std::string, ConfigurationType> mConfigTypeMap;
   std::map<std::string, bool> mBooleanConfigMap;
   std::map<std::string, int> mIntegerConfigMap;
   std::map<std::string, unsigned int> mUnsignedIntConfigMap;
   std::map<std::string, std::string> mStringConfigMap;
   std::map<std::string, IntOptional> mIntOptionalConfigMap;
}; // end class Configuration

} // end namespace PV

#endif // CONFIGURATION_HPP_
