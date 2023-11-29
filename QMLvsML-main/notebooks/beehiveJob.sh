#!/bin/sh  
#  
# myscript.sh  
#  
module add compiler/intel  
source /etc/profile.d/modules.sh
module add soft/anaconda3/latest

echo "starting now..."  
date  
  
/p/slg/bin/computejob -v -d /p/slg/data/computejob.in  
 # Activate the conda environment
conda activate /m/soft/miniconda/py310_23.3.1-0
 # Run the Python script
python /u/a/abbassi/Downloads/QMLvsML-main/QMLvsML-main/notebooks/4.b-exact_inversion.py 
echo "finished."  
date  
