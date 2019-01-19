import argparse
import glob
import os
import sys

import cv2
import pandas as pd
import numpy as np

class_img = ["BLD", "BYC", "COW", "CTR", "FLW", "TRE"]

def read_images_list(image_dir):
    file_list = dict()
    for clss in class_img:
        filelist = glob.glob(image_dir+"/*" + clss + "*")
        filelist = sorted(filelist)
        clss_name = filelist[0].split(" ")[0].split("/")[1].split("_")[1]
        file_list[clss_name] = filelist

    return file_list


def read_images(images_files):
    images_list = dict()
    images_list_gray = dict()
    for keyname in images_files:
        img_list = []
        img_list_gray = []
        for i, img_path in enumerate(images_files[keyname]):
            img = cv2.imread(img_path)
            img_list_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            img_list.append(img)
        img_list = np.array(img_list)
        img_list_gray = np.array(img_list_gray)
        images_list[keyname] = img_list
        images_list_gray[keyname] = img_list_gray
    return images_list, images_list_gray


def main():
    parser = argparse.ArgumentParser(
        description='A script to make clustering or use cluster to queries')
    parser.add_argument('--database_dir', '-d',
                        type=str, default="database", help="training images directory")
    parser.add_argument('--rlt_dir', '-rlt',
                        type=str, default="", help="rlt files directory")
    parser.add_argument('--queries_dir', '-q',
                        type=str, default="queries", help="queries images directory")
    parser.add_argument('--output_dir', '-o',
                        type=str, default="output_queries", help="outputs of queried result directory")

    args = parser.parse_args()

    db_img = args.database_dir
    query_img = args.queries_dir
    rlt_dir = args.rlt_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print("selected output directory existed",output_dir)
        exit(1)

    db_img = read_images_list(db_img)
    query_img = read_images_list(query_img)
    rlt_dir = sorted(glob.glob(rlt_dir+"/*"))
    for model_rlt in rlt_dir:
        print("model_rlt:",model_rlt)
        query_path = os.path.join(output_dir,model_rlt.split("/")[1])
        all_queried_rlt = sorted(glob.glob(model_rlt+"/*"))
        print("query_path:",query_path)
        os.mkdir(query_path)

        for rlt in all_queried_rlt:
            print(rlt)
            tmp_path = rlt.split("/")[-1].split(".")[0].replace("(","\(").replace(")","\)")
            path = os.path.join(query_path,tmp_path)
            print(path)

            os.system("mkdir {}".format(path))
            data = []
            with open(rlt,"r") as f:
                for i in range(0,11):
                    tmp = f.readline().strip().split(" ")
                    data.append(tmp)
                data = data[1:]
            queries = []
            queries_path = []
            for d in data:
                queries_path.append(os.path.join(query_path,d[0]))
                tmp_name1 = d[0].split("(")[0]
                tmp_name2 = d[0].split("(")[1].split(")")[0]
                tmp = tmp_name1+"\ \("+tmp_name2+"\)"+".JPG"
                queries.append([tmp])

            for index,qpath in enumerate(queries_path):
                for q in queries[index]:
                    os.system("cp {}/{} {}/{}_{}".format(args.database_dir,q,path,index+1,q))


if __name__ == "__main__":
    main()