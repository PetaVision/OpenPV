$(BUILDDIR)/CLBuffer.o: $(SRCDIR)/arch/opencl/CLBuffer.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/CLDevice.o: $(SRCDIR)/arch/opencl/CLDevice.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/CLKernel.o: $(SRCDIR)/arch/opencl/CLKernel.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/HyPerCol.o: $(SRCDIR)/columns/HyPerCol.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/HyPerColRunDelegate.o: $(SRCDIR)/columns/HyPerColRunDelegate.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/Communicator.o: $(SRCDIR)/columns/Communicator.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/InterColComm.o: $(SRCDIR)/columns/InterColComm.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/DataStore.o: $(SRCDIR)/columns/DataStore.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/HyPerLayer.o: $(SRCDIR)/layers/HyPerLayer.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/Gratings.o: $(SRCDIR)/layers/Gratings.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/Image.o: $(SRCDIR)/layers/Image.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/LayerDataInterface.o: $(SRCDIR)/layers/LayerDataInterface.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/Movie.o: $(SRCDIR)/layers/Movie.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/HyPerConn.o: $(SRCDIR)/connections/HyPerConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/GaborConn.o: $(SRCDIR)/connections/GaborConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/CocircConn.o: $(SRCDIR)/connections/CocircConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/InhibConn.o: $(SRCDIR)/connections/InhibConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/KernelConn.o: $(SRCDIR)/connections/KernelConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/PoolConn.o: $(SRCDIR)/connections/PoolConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/RandomConn.o: $(SRCDIR)/connections/RandomConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/RuleConn.o: $(SRCDIR)/connections/RuleConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/Retina.o: $(SRCDIR)/layers/Retina.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/LIF.o: $(SRCDIR)/layers/LIF.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/PVParams.o: $(SRCDIR)/io/PVParams.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/ConnectionProbe.o: $(SRCDIR)/io/ConnectionProbe.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/GLDisplay.o: $(SRCDIR)/io/GLDisplay.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/LayerProbe.o: $(SRCDIR)/io/LayerProbe.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/LinearActivityProbe.o: $(SRCDIR)/io/LinearActivityProbe.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/PointProbe.o: $(SRCDIR)/io/PointProbe.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/PostConnProbe.o: $(SRCDIR)/io/PostConnProbe.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/StatsProbe.o: $(SRCDIR)/io/StatsProbe.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/param_parser.o: $(SRCDIR)/io/parser/param_parser.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/PVConnection.o: $(SRCDIR)/connections/PVConnection.c $(HEADERS)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILDDIR)/PVLayer.o: $(SRCDIR)/layers/PVLayer.c $(HEADERS)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILDDIR)/fileread.o: $(SRCDIR)/layers/fileread.c $(HEADERS)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILDDIR)/io.o: $(SRCDIR)/io/io.c $(HEADERS)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILDDIR)/fileio.o: $(SRCDIR)/io/fileio.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/imageio.o: $(SRCDIR)/io/imageio.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/tiff.o: $(SRCDIR)/io/tiff.c $(HEADERS)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILDDIR)/param_lexer.o: $(SRCDIR)/io/parser/param_lexer.c $(HEADERS)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILDDIR)/conversions.o: $(SRCDIR)/utils/conversions.c $(HEADERS)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILDDIR)/box_muller.o: $(SRCDIR)/utils/box_muller.c $(HEADERS)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILDDIR)/cl_random.o: $(SRCDIR)/utils/cl_random.c $(HEADERS)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILDDIR)/Timer.o: $(SRCDIR)/utils/Timer.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<
