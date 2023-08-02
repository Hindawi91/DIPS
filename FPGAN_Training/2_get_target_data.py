import os, sys, shutil

target_directory = #Target DS directory e.g. "target_DS_directory/"

sub_dir = ['train','val','test']

for sub in sub_dir:  
    translated_directory = f"target_directory{sub}/"
    
    if sub == "val":
        translated_target_directory = "./data/brats/syn_val/test/positive/"
    else:
        translated_target_directory = f"./data/brats/syn/{sub}/positive/"



    translated_img_folders = os.listdir(translated_directory)
    number_of_translated_img_folders = len(os.listdir(translated_directory))

    # Make sure base_target_directory is empty
    print(f"/n removing old images from translated_target_directory")
    shutil.rmtree(translated_target_directory)
    if not os.path.exists(translated_target_directory):
        os.makedirs(translated_target_directory)
    print(f"/n translated_target_directory is now empty") 
    
    # ----------------------------------------------  
    
    for i,folder in enumerate(translated_img_folders):
        
        img_folder_directory = f"{translated_directory}{folder}/"
        
        for img in os.listdir(img_folder_directory):
    
            img_dir = f"{img_folder_directory}{img}"
            shutil.copy(img_dir, translated_target_directory)
    
        
        print(f" ----------- sub folder: {sub} ---- {i} out of {number_of_translated_img_folders} moved ------------")



    
