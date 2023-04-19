#include "PVAssert.hpp"
#include "utils/PathComponents.hpp" // baseName
#include <stdarg.h>
#include <stdio.h>

namespace PV {

void pv_abort_message(const char *file, int line, const char *fmt, ...) {
   getOutputStream().flush();
   std::ostream &st    = getErrorStream();
   static int buf_size = 1024;
   char msg[buf_size];
   va_list args;
   va_start(args, fmt);
   vsnprintf(msg, buf_size, fmt, args);
   va_end(args);
   st << "assert failed";
   if (file) {
      std::string filebasename = baseName(file);
      st << " <" << filebasename << ":" << line << ">";
   }
   st << ": " << msg;
   st.flush();
   abort();
}

void pv_assert_failed(const char *file, int line, const char *condition) {
   pv_abort_message(file, line, "%s\n", condition);
}

void pv_assert_failed_message(
      const char *file,
      int line,
      const char *condition,
      const char *fmt,
      ...) {
   /* Build up custom error string */
   va_list args;
   va_start(args, fmt);
   static int buf_size = 1024;
   char msg[buf_size];
   vsnprintf(msg, buf_size, fmt, args);
   va_end(args);

   pv_abort_message(file, line, "%s: %s\n", condition, msg);
}

void print_stacktrace(FILE *out, unsigned int max_frames) {
   StackTrace() << "stack trace:" << std::endl;

   // storage array for stack trace address data
   void *addrlist[max_frames + 1];

   // retrieve current stack addresses
   int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void *));

   if (addrlen == 0) {
      StackTrace() << "  <empty, possibly corrupt>" << std::endl;
      return;
   }

   // resolve addresses into strings containing "filename(function+address)",
   // this array must be free()-ed
   char **symbollist = backtrace_symbols(addrlist, addrlen);

   // allocate string which will be filled with the demangled function name
   size_t funcnamesize = 256;
   char *funcname      = (char *)malloc(funcnamesize);

   // iterate over the returned symbol lines. skip the first, it is the
   // address of this function.
   for (int i = 1; i < addrlen; i++) {
      char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

      // find parentheses and +address offset surrounding the mangled name:
      // ./module(function+0x15c) [0x8048a6d]
      for (char *p = symbollist[i]; *p; ++p) {
         if (*p == '(')
            begin_name = p;
         else if (*p == '+')
            begin_offset = p;
         else if (*p == ')' && begin_offset) {
            end_offset = p;
            break;
         }
      }

      if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
         *begin_name++   = '\0';
         *begin_offset++ = '\0';
         *end_offset     = '\0';

         // mangled name is now in [begin_name, begin_offset) and caller
         // offset in [begin_offset, end_offset). now apply
         // __cxa_demangle():

         int status;
         char *ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
         if (status == 0) {
            funcname = ret; // use possibly realloc()-ed string
            StackTrace() << "  " << symbollist[i] << " : " << funcname << "+" << begin_offset
                         << std::endl;
         }
         else {
            // demangling failed. Output function name as a C function with
            // no arguments.
            StackTrace() << "  " << symbollist[i] << " : " << begin_name << "+" << begin_offset
                         << std::endl;
         }
      }
      else {
         // couldn't parse the line? print the whole line.
         StackTrace() << "  " << symbollist[i] << std::endl;
      }
   }

   free(funcname);
   free(symbollist);
}

} // end namespace PV
