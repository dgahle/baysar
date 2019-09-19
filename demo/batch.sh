#! /bin/bash
#$ -pe smp 8
#$ -m ea 
#$ -M daljeet.gahle@strath.ac.uk

cd /home/dgahle/baysar/demo/

source /home/dgahle/bin/start_cherab_env.sh

ipython baysar_asdex_demo.py