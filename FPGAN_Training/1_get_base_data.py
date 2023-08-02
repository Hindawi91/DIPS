import os, sys, shutil

base_directory =  #Source DS directory e.g. "source_DS_directory/train/"
base_target_directory = "./data/brats/syn/train/negative/"

base_img_folders = os.listdir(base_directory)
number_of_base_img_folders = len(os.listdir(base_directory))

# Make sure base_target_directory is empty
print(f"/n removing old images from base_target_directory")
shutil.rmtree(base_target_directory)
if not os.path.exists(base_target_directory):
    os.makedirs(base_target_directory)
print(f"/n base_target_directory is now empty") 

# ----------------------------------------------  

for i,folder in enumerate(base_img_folders):
    
    img_folder_directory = f"{base_directory}{folder}/"
    
    for img in os.listdir(img_folder_directory):

        img_dir = f"{img_folder_directory}{img}"
        shutil.copy(img_dir, base_target_directory)

    
    print(f" ----------- {i} out of {number_of_base_img_folders} moved ------------")




    
