#Return files provided for out of sample submission have different naming conventions - lets massage the file names so
#that CSVDataLoader classes can load both in sample and out of sample data seamlessly

import shutil
import glob
import os
import re
import logging
import csv

BASE_PATH = 'c:/users/orduk/PycharmProjects/AT2024Port/data/outofsample/'

def insertHeaders():
    for f in glob.glob(BASE_PATH+'/Returns/Returns*c.csv'):
        logging.debug('Processing %s',f)
        with open(f, 'r') as original: data = original.read()
        if 'CUSIP' not in data:
            with open(f, 'w', newline='') as modified:
                writer = csv.writer(modified)
                writer.writerow(['CUSIP','Return',''])
                modified.write(data)
def move_rename_files():
    for f in glob.glob(BASE_PATH+'Returns/*/sret*c.csv'):
        new_dir = os.path.sep.join(os.path.abspath(os.path.dirname(f)).split(os.path.sep)[0:-1])
        fname = os.path.basename(f)
        match = re.search(r'(\d{6})', fname)
        if not match:
            logging.warning('Skipping %s',f)
            continue
        newFname = str.replace(fname,'sret','Returns20')

        src = os.path.abspath(f)
        dest = os.path.abspath(os.path.join(new_dir,newFname))
        print('Moving ',src,' to ',dest)
        shutil.move(src,dest)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    move_rename_files()
    insertHeaders()
    print('Done')