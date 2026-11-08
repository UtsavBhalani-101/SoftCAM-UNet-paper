import os, pandas, numpy as np, argparse, pandas as pd, shutil
from PIL import Image
from pydicom import dcmread
import json


def get_args_parser():
    parser = argparse.ArgumentParser('args script', add_help=False)
    parser.add_argument('--csv-path', default='/home/ccchen/projects/ich_preprocessing/training/utils_dir/stage_2_sort_by_PatientID_and_zposition_dataloader.csv', type=str, help='final_csv_path')
    parser.add_argument('--save-dir', default='/home/ccchen/projects/dr_hsieh/kaggle_train_0112_png_3channels', type=str, help='save_dir')
    parser.add_argument('--dcm-dir', default='/media/ccchen/新增磁碟區/abc/stage_2_train', type=str, help='dcm_dir')
    parser.add_argument('--id_json', default='./select_ich.json', type=str, help='dcm_dir')


def get_single_dir(df,uid,args):
    target_dir = args.save_dir
    sub_df = df.loc[df['StudyInstanceUID'] == uid]
    final_dir = os.path.join(target_dir,uid)
    final_png_dir = os.path.join(png_dir,uid)
    os.makedirs(final_dir, exist_ok = True)
    os.makedirs(final_png_dir, exist_ok = True)
    id_list = sub_df['ID'].tolist()

    for i in range(len(id_list)):
        dcm_name = os.path.join(args.dcm_dir,id_list[i]+'.dcm')
        target_name = os.path.join(final_dir,str(i+1)+'_'+id_list[i]+'.dcm')
        
        
        shutil.copy(dcm_name,target_name)

def get_list_dirs(df,id_list,args):
    start = time.time()
    for i in range(len(id_list)):
        get_single_dir(df,id_list[i],args)
    
    end =  time.time()
    print("Done.", end - start)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('args script', parents=[get_args_parser()])
    args = parser.parse_args()
   
    df = pd.read_csv(args.csv_path)
    id_dict = json.load(open(args.id_json , 'rb'))


    
                    
    dirlist = set([ key for key in id_dict])
    key_list =df['StudyInstanceUID'].apply(lambda x : x in dirlist)
    df = df[key_list]
    get_list_dirs(df, list(dirlist),args)

    

