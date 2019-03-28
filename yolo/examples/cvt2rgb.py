import glob

for filename in glob.glob("../MSCOCO/val2017/*.jpg"):
    inp_image = imread(filename)
    [...]
