#!/bin/bash

path=`pwd`
processor_num=`cat /proc/cpuinfo | grep processor | grep -v grep | wc -l`

###default use wino 2x3
if [ ! -n "$1" ] ;then
    cmd="make -j${processor_num}"
    bin=winograd_conv_2x3
    lib_path=$path/libs/hklib_wino_2x3/lib64/
elif [ $1 == wino_2x3 ]; then
    cmd="make -j${processor_num} WINO_2x3=TRUE"
    bin=winograd_conv_2x3
    lib_path=$path/libs/hklib_wino_2x3/lib64/
elif [ $1 == wino_4x3 ]; then
    cmd="make -j${processor_num} WINO_4x3=TRUE"
    bin=winograd_conv_4x3
    lib_path=$path/libs/hklib_wino_4x3/lib64/
else
    cmd="make -j${processor_num}"
    bin=winograd_conv_2x3
    lib_path=$path/libs/hklib_wino_2x3/lib64/
fi

rm -f ./$bin
eval "$cmd" > /dev/null 2>&1

export LD_LIBRARY_PATH=$lib_path:$LD_LIBRARY_PATH
./$bin
export -n LD_LIBRARY_PATH
