SRCDIR = ../src
BUILDDIR = build
GDALDIR = /usr/local
LIBPV = ../lib/libpv.a

INPARAMS = input/params.test

CPP = mpic++
CC  = mpicc
#CPP = c++
#CC  = cc

CFLAGS = -g -std=c99 -Wall -I../src
CPPFLAGS = -g -Wall -I../src

LIBS = $(LIBPV) -L$(GDALDIR)/lib -lgdal

TESTS = test_kg test_kxpos test_kypos test_pos \
        test_nearby_neighbor test_delta_pos test_extend_border \
        test_patch_head test_delta test_sign test_borders \
        test_datatypes test_post_weights test_mirror_BCs \
        test_border_activity

MPI_TESTS = test_read_header test_image_io test_weights_io

DEPRECATED_TESTS = test_decode_msg \
                   test_layer_direct     # fixme

.SUFFIXES: .cpp .c .o

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $(@:.o=.cpp)

.c.o:
	$(CC) $(CFLAGS) -c $(@:.o=.c)

$(BUILDDIR)/%.o: %.cpp
	@echo 'Building file: $<'
	$(CPP) $(CPPFLAGS) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

$(BUILDDIR)/%.o: %.c
	@echo 'Building file: $<'
	$(CC) $(CFLAGS) -c -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

all: make-in-subdirs $(TESTS) $(MPI_TESTS)

check: runtests

test_read_header: $(BUILDDIR)/test_read_header.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_layer_direct: $(BUILDDIR)/test_layer_direct.o $(BUILDDIR)/TestLayer.o $(PVLIB)
	$(CPP) -o $@ $< $(BUILDDIR)/TestLayer.o $(LIBS)

test_image_io: $(BUILDDIR)/test_image_io.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_weights_io: $(BUILDDIR)/test_weights_io.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_kg: $(BUILDDIR)/test_kg.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_kxpos: $(BUILDDIR)/test_kxpos.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_kypos: $(BUILDDIR)/test_kypos.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_nearby_neighbor: $(BUILDDIR)/test_nearby_neighbor.o  $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_delta_pos: $(BUILDDIR)/test_delta_pos.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_pos: $(BUILDDIR)/test_pos.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_post_weights: $(BUILDDIR)/test_post_weights.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_decode_msg: $(BUILDDIR)/test_decode_msg.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_extend_border: $(BUILDDIR)/test_extend_border.o $(PVLIB)
	$(CC) -o $@ $< $(LIBS)

test_borders: $(BUILDDIR)/test_borders.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_border_activity: $(BUILDDIR)/test_border_activity.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_mirror_BCs: $(BUILDDIR)/test_mirror_BCs.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_datatypes: $(BUILDDIR)/test_datatypes.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_patch_head: $(BUILDDIR)/test_patch_head.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_delta: $(BUILDDIR)/test_delta.o $(PVLIB)
	$(CC) -o $@ $<

test_sign: $(BUILDDIR)/test_sign.o $(PVLIB)
	$(CC) -o $@ $<

runMPItests: make-in-subdirs $(MPI_TESTS)
	@rm -rf output/*.pvp
	@echo "Running test_read_header with 1 node ..."
	@test_read_header -n 1 -p $(INPARAMS)
	@echo "Running test_read_header with 2 nodes ..."
	@mpirun -np 2 test_read_header -n 1 -p $(INPARAMS)
	@echo "Running test_read_header with 4 nodes ..."
	@mpirun -np 4 test_read_header -n 1 -p $(INPARAMS)
	@rm -rf output/*.pvp
	@rm -rf output/*.tif
	@echo "Running test_image_io with 1 node ..."
	@test_image_io -n 1 -p $(INPARAMS)
	@echo "Running test_image_io with 2 nodes ..."
	@mpirun -np 2 test_image_io -n 1 -p $(INPARAMS)
	@echo "Running test_image_io with 4 nodes ..."
	@mpirun -np 4 test_image_io -n 1 -p $(INPARAMS)
	@echo "Running test_image_io with 6 nodes ..."
	@mpirun -np 6 test_image_io -n 1 -p $(INPARAMS)
	@echo "Running test_weights_io with 1 node ..."
	@test_weights_io -n 1 -p $(INPARAMS)
	@echo "Running test_weights_io with 2 nodes ..."
	@mpirun -np 2 test_weights_io -n 1 -p $(INPARAMS)
	@echo "Running test_weights_io with 4 nodes ..."
	@mpirun -np 4 test_weights_io -n 1 -p $(INPARAMS)
	@echo "Running test_weights_io with 6 nodes ..."
	@mpirun -np 6 test_weights_io -n 1 -p $(INPARAMS)

runtests: make-in-subdirs $(TESTS)
	@rm -rf output/*.pvp
	@echo "Running test_kg ..."
	@./test_kg
	@echo "Running test_kxpos ..."
	@./test_kxpos
	@echo "Running test_kypos ..."
	@./test_kypos
	@echo "Running test_nearby_neighbor ..."
	@./test_nearby_neighbor
	@echo "Running test_patch_head ..."
	@./test_patch_head
	@echo "Running test_post_weights ..."
	@./test_post_weights -p input/params.test
#	@echo "Running test_pos ..."
#	@test_pos
	@echo "Running test_delta_pos ..."
	@./test_delta_pos
	@echo "Running test_extend_border ..."
	@./test_extend_border
	@echo "Running test_patch_head ..."
	@./test_patch_head
	@echo "Running test_delta ..."
	@./test_delta
	@echo "Running test_sign ..."
	@./test_sign
	@echo "Running test_mirror_BCs ..."
	@./test_mirror_BCs -p $(INPARAMS)
	@echo "Running test_border_activity ..."
	@./test_border_activity -p $(INPARAMS)
#	@echo "Running test_borders ..."
#	@test_borders -p $(INPARAMS)
#	@echo "Running test_layer_direct with 1 node ..."
#	@test_layer_direct -p $(INPARAMS)

make-in-subdirs: force
	@for subdir in $(SUBDIRS); do \
	    if [ -d $$subdir -a -f $$subdir/Makefile ] ; then \
	        echo "making in $$subdir" ; \
	        (cd $$subdir && $(MAKE) all) ; \
	    fi \
	done

clean: clean-in-subdirs
	rm -f *.o *.bin output/*.pvp
	rm -f $(TESTS) $(MPI_TESTS) *.o *.bin output/*.pvp
	rm -f $(BUILDDIR)/*.o

cleanall: clean
	rm -f $(MPI_OBJ)
	rm -rf output

clean-in-subdirs: force
	@for subdir in $(SUBDIRS); do \
	    if [ -d $$subdir -a -f $$subdir/Makefile ] ; then \
	        echo "cleaning in $$subdir" ; \
	        (cd $$subdir && $(MAKE) clean) ; \
	    fi \
	done

force:
