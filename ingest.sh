#!/bin/sh -Tf
#PBS -N WL-INGEST
#PBS -l nodes=1
#PBS -l walltime=12:00:00
#PBS -l pmem=6gb
#PBS -k oe
#PBS -j oe
#PBS -v PATH,WEAVEIO_USER,WEAVEIO_PASSWORD

# This should be run as:
# qsub -V -v NIGHT=20170226 ingest.sh
# or to submit jobs for lots of nights:
# ls $WEAVEIO_ROOTDIR/raw/ | grep -v txt | xargs -I{} qsub -V -v NIGHT={} ingest.sh

export WEAVEIO_DB=lorentz

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo $PATH
SAVED_WEAVEIO_USER=$WEAVEIO_USER
SAVED_WEAVEIO_PASSWORD=$WEAVEIO_PASSWORD
export PYTHONUNBUFFERED=1
cd /home2/bamford/weave-io
echo Starting ingest `date`
echo Ingesting night $NIGHT to DB $WEAVEIO_DB
conda run -n weaveio WEAVEIO_USER=$SAVED_WEAVEIO_USER WEAVEIO_PASSWORD=$SAVED_WEAVEIO_PASSWORD python ingest.py $NIGHT
echo ------------------------------------------------------
echo Job ends `date`
echo ------------------------------------------------------
echo ------------------------------------------------------
