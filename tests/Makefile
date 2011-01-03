SRCDIR = ../src
BUILDDIR = build
GDALDIR = /opt/local/lib 
#GDALDIR = /usr/local
LIBPV = ../lib/libpv.a

INPARAMS = input/params.test

CPP = mpic++
CC  = mpicc
#CPP = c++
#CC  = cc

CFLAGS = -g -std=c99 -Wall -I../src
CPPFLAGS = -g -Wall -I../src

#LIBS = $(LIBPV) -L$(GDALDIR)/lib -lgdal
LIBS = $(LIBPV) -L$(GDALDIR) -lgdal

CL_LD_FLAGS = -framework OpenCL

TESTS = test_kg test_kxpos test_kypos test_pos \
        test_nearby_neighbor test_delta_pos test_extend_border \
        test_patch_head test_delta test_sign test_borders \
        test_datatypes test_post_weights test_gauss2d test_cocirc test_mirror_BCs \
        test_border_activity test_cl_random test_cl_lif

MPI_TESTS = #test_read_header test_image_io test_weights_io test_constant_input

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

test_constant_input: $(BUILDDIR)/test_constant_input.o $(BUILDDIR)/TestImage.o $(PVLIB)
	$(CPP) -o $@ $< $(BUILDDIR)/TestImage.o $(LIBS)

test_image_io: $(BUILDDIR)/test_image_io.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_cl_random: $(BUILDDIR)/test_cl_random.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS) $(CL_LD_FLAGS)

test_cl_lif: $(BUILDDIR)/test_cl_lif.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS) $(CL_LD_FLAGS)

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

test_gauss2d: $(BUILDDIR)/test_gauss2d.o $(PVLIB)
	$(CPP) -o $@ $< $(LIBS)

test_cocirc: $(BUILDDIR)/test_cocirc.o $(PVLIB)
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
	@./test_read_header -n 1 -p $(INPARAMS)
	@echo "Running test_read_header with 2 nodes ..."
	@mpirun -np 2 test_read_header -n 1 -p $(INPARAMS)
	@echo "Running test_read_header with 4 nodes ..."
	@mpirun -np 4 test_read_header -n 1 -p $(INPARAMS)
	@rm -rf output/*.pvp
	@rm -rf output/*.tif
	@echo "Running test_image_io with 1 node ..."
	@./test_image_io -n 1 -p $(INPARAMS)
	@echo "Running test_image_io with 2 nodes ..."
	@mpirun -np 2 test_image_io -n 1 -p $(INPARAMS)
	@echo "Running test_image_io with 4 nodes ..."
	@mpirun -np 4 test_image_io -n 1 -p $(INPARAMS)
#	@echo "Running test_image_io with 6 nodes ..."
#	@mpirun -np 6 test_image_io -n 1 -p $(INPARAMS)
	@echo "Running test_weights_io with 1 node ..."
	@./test_weights_io -n 1 -p $(INPARAMS)
	@echo "Running test_weights_io with 2 nodes ..."
	@mpirun -np 2 test_weights_io -n 1 -p $(INPARAMS)
	@echo "Running test_weights_io with 4 nodes ..."
	@mpirun -np 4 test_weights_io -n 1 -p $(INPARAMS)
#	@echo "Running test_weights_io with 6 nodes ..."
#	@mpirun -np 6 test_weights_io -n 1 -p $(INPARAMS)
	@echo "Running test_constant_input with 2 nodes ..."
	@mpirun -np 2 test_constant_input -p $(INPARAMS)
	@echo "Running test_constant_input with 4 nodes ..."
	@mpirun -np 4 test_constant_input -p $(INPARAMS)

runtests: make-in-subdirs $(TESTS)
	@rm -rf output/*.pvp
	@echo "Running test_kg ..."
	@if ./test_kg; then echo "test_kg passed."; else echo "test_kg failed."; fi
	@echo "Running test_kxpos ..."
	@if ./test_kxpos; then echo "test_kxpos passed."; else echo "test_kxpos failed."; fi
	@echo "Running test_kypos ..."
	@if ./test_kypos; then echo "test_kypos passed."; else echo "test_kypos failed."; fi
	@echo "Running test_nearby_neighbor ..."
	@if ./test_nearby_neighbor; then echo "test_nearby_neighbor passed."; else echo "test_nearby_neighbor failed."; fi
	@echo "Running test_patch_head ..."
	@if ./test_patch_head; then echo "test_patch_head passed."; else echo "test_patch_head failed."; fi
	@echo "Running test_post_weights ..."
	@if ./test_post_weights -p input/params.test; then echo "test_post_weights passed."; else echo "test_post_weights failed."; fi
	@echo "Running test_gauss2d ..."
	@if ./test_gauss2d -p input/params.test; then echo "test_gauss2d passed."; else echo "test_gauss2d failed."; fi
	@echo "Running test_cocirc ..."
	@if ./test_cocirc -p input/params.test; then echo "test_cocirc passed."; else echo "test_cocirc failed."; fi
#	@echo "Running test_pos ..."
#	@if ./test_pos; then echo "test_pos passed."; else echo "test_pos failed."; fi
	@echo "Running test_delta_pos ..."
	@if ./test_delta_pos; then echo "test_delta_pos passed."; else echo "test_delta_pos failed."; fi
	@echo "Running test_extend_border ..."
	@if ./test_extend_border; then echo "test_extend_border passed."; else echo "test_extend_border failed."; fi
	@echo "Running test_patch_head ..."
	@if ./test_patch_head; then echo "test_patch_head passed."; else echo "test_patch_head failed."; fi
	@echo "Running test_delta ..."
	@if ./test_delta; then echo "test_delta passed."; else echo "test_delta failed."; fi
	@echo "Running test_sign ..."
	@if ./test_sign; then echo "test_sign passed."; else echo "test_sign failed."; fi
	@echo "Running test_mirror_BCs ..."
	@if ./test_mirror_BCs -p $(INPARAMS); then echo "test_mirror_BCs passed."; else echo "test_mirror_BCs passed."; fi
	@echo "Running test_border_activity ..."
	@if ./test_border_activity -p $(INPARAMS); then echo "test_border_activity passed."; else echo "test_border_activity failed"; fi
#	@echo "Running test_constant_input with 1 node ..."
#	@if ./test_constant_input -p $(INPARAMS) then echo "test_constant_input passed."; else echo "test_constant_input failed"; fi
#	@echo "Running test_borders ..."
#	@if test_borders -p $(INPARAMS) then echo "test_constant_input passed."; else echo "test_constant_input failed"; fi

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
