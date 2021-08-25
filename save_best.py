import os, shutil, glob

for h5 in glob.glob('/uctgan/data/ray_results/AAE_uct_test2/*/*/*h5*'):
    print("Find : ", h5)
    os.makedirs(h5.replace("AAE_uct_test2", "saved"), exist_ok=True)
    shutil.copy2(h5, 
        h5.replace("AAE_uct_test2", "saved"))
        