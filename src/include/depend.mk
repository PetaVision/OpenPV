$(BUILDDIR)/HyPerCol.o: $(SRCDIR)/columns/HyPerCol.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/InterColComm.o: $(SRCDIR)/columns/InterColComm.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/DataStore.o: $(SRCDIR)/columns/DataStore.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/HyPerLayer.o: $(SRCDIR)/layers/HyPerLayer.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/HyPerConn.o: $(SRCDIR)/connections/HyPerConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/GaborConn.o: $(SRCDIR)/connections/GaborConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/CocircConn.o: $(SRCDIR)/connections/CocircConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/InhibConn.o: $(SRCDIR)/connections/InhibConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/LongRangeConn.o: $(SRCDIR)/connections/LongRangeConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/PoolConn.o: $(SRCDIR)/connections/PoolConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/RandomConn.o: $(SRCDIR)/connections/RandomConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/RuleConn.o: $(SRCDIR)/connections/RuleConn.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/Example.o: $(SRCDIR)/layers/Example.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/LIF2.o: $(SRCDIR)/layers/LIF2.c $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/Retina.o: $(SRCDIR)/layers/Retina.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/LGN.o: $(SRCDIR)/layers/LGN.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/V1.o: $(SRCDIR)/layers/V1.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/PVParams.o: $(SRCDIR)/io/PVParams.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/ConnectionProbe.o: $(SRCDIR)/io/ConnectionProbe.cpp $(HEADERS)
	$(CPP) -c $(CPPFLAGS) -o $@ $<

$(BUILDDIR)/PVLayerProbe.o: $(SRCDIR)/io/PVLayerProbe.cpp $(HEADERS)
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

$(BUILDDIR)/tiff.o: $(SRCDIR)/io/tiff.c $(HEADERS)
	$(CC) -c $(CFLAGS) -o $@ $<

$(BUILDDIR)/param_lexer.o: $(SRCDIR)/io/parser/param_lexer.c $(HEADERS)
	$(CC) -c $(CFLAGS) -o $@ $<
