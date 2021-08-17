#CC=/usr/bin/gcc-7
CC=g++
path :=$(shell pwd)

ifeq ($(WINO_4x3), TRUE)
wino_dir=hklib_wino_4x3
out_bin=winograd_conv_4x3
else
wino_dir=hklib_wino_2x3
out_bin=winograd_conv_2x3
endif

###Hygonblis
LAdir        = $(path)/libs/$(wino_dir)
LAinc        = -I$(LAdir)/include
LAlib        = -L$(LAdir)/lib64/ -ldnnl
#LAlib        = -L/home/liulei/work/HgDNN/build/src/ -ldnnl

CFLAGS      += $(LAinc) -O3
LDFLAGS     += $(LAlib)


all:
	$(CC) $(CFLAGS) winograd_conv.cpp -o $(out_bin) $(LDFLAGS)

clean:
	rm $(out_bin)
