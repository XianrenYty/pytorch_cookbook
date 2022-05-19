import os
import os.path as osp
import glob

base = os.getcwd()

dirs = glob.glob(os.path.join(base, 'source', 'chapter*'))

for dir in dirs:
    # for nb in glob.glob(os.path.join(dir, '*.ipynb')):
    #     f_name  = os.path.basename(nb)
    #     f_dir = nb.rsplit("\\")[-2] 
    #     print(f_dir + '/' + f_name)
    for md in glob.glob(os.path.join(dir, '*.md')):
        f_name  = os.path.basename(md)[:-3]
        f_dir = md.rsplit("\\")[-2] 
        print(f_dir + '/' + f_name)
