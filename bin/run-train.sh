#!/bin/bash
CNFHOME="/home/zywang/work/raptorx-ss8-src";
if [ $# -lt 1 ] ; then 
# This script is used to generate training file from a list of PDB files with known secondary structures.
exit -1
fi
procname=$$
trainfile=trainfile.$$
validfile=validfile.$$
tempresultfile=.tmprsfile.$$.log
$CNFHOME/bin/GenFeat.train.pl $1 $trainfile &> .genfeat_train.$procname.log
$CNFHOME/bin/GenFeat.train.pl $2 $validfile &> .genfeat_valid.$procname.log


$CNFHOME/bin/bcnf_mpitp.train CONF $CNFHOME/data/CNF.ax.train.conf TRAIN $trainfile TEST $validfile RESULT $tempresultfile &> train.$procname.log

# output

rm .*.$procname.log

