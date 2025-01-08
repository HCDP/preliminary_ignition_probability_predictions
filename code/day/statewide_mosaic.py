import os
import sys
import subprocess

MASTER_DIR = os.environ.get("PROJECT_ROOT")
OUTPUT_DIR = MASTER_DIR + "data_outputs/day/tiff/"

def statewide_mosaic():
    icode_list = ['bi','ka','mn','oa']
    file_names = [OUTPUT_DIR+'Probability_'+icode+'.tif' for icode in icode_list]
    output_name = OUTPUT_DIR + 'Probability_statewide.tif'
    cmd = "gdal_merge.py -o "+output_name+" -of gtiff -co COMPRESS=LZW -n -9999 -a_nodata -9999"
    return subprocess.run(cmd.split()+file_names).returncode

#should run date agnostic. Will automatically mosaic the existing county files in the container
if __name__=="__main__":
    rtn = statewide_mosaic()
    if rtn == 0:
        print("Mosaic completed")
    else:
        print("Mosaic failed. Check files.")