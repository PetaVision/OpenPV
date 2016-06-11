/*
 * FileStream.hpp
 *
 *  Created on: Jun 9, 2016
 *      Author: pschultz
 */

#ifndef SRC_IO_FILESTREAM_HPP_
#define SRC_IO_FILESTREAM_HPP_

#include <ostream>
#include <fstream>

namespace PV {

class OutStream {
public:
   OutStream(std::ostream& stream);
   virtual ~OutStream() {}
   std::ostream& outStream() { return *mOutStream; }

protected:
   OutStream() {}
   void setOutStream(std::ostream& stream);

private:

// Data members
private:
   std::ostream * mOutStream = nullptr;
};

class FileStream : public OutStream {
public:
   FileStream(char const * path, std::ios_base::openmode mode, bool verifyWrites = false);
   virtual ~FileStream();
   bool readable() { return mMode & std::ios_base::out; }
   bool writeable() { return mMode & std::ios_base::in; }
   bool readwrite() { return readable() && writeable(); }
   std::istream& inStream() { return *mInStream; }

protected:
   FileStream() {}
   void initialize(char const * path, std::ios_base::openmode mode, bool verifyWrites);

private:
   void openFile(char const * path, std::ios_base::openmode mode);
   void closeFile();

// Data members
private:
   char * mPath = nullptr;
   std::ios_base::openmode mMode = std::ios_base::out;
   std::fstream * mStrPtr = nullptr;
   std::istream * mInStream = nullptr;
   std::streambuf::pos_type mFilePos = (std::streambuf::pos_type) 0;
   std::streambuf::pos_type mFileLength = (std::streambuf::pos_type) 0;
   bool mVerifyWrites = false;
   int const mMaxAttempts = 5;
};

} /* namespace PV */

#endif /* SRC_IO_FILESTREAM_HPP_ */
