import argparse
import glob
import logging
import logging.handlers as handlers
import os
import sys
import time

import cv2
import coloredlogs
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib


def setup_logger():
    logging.StreamHandler(sys.stderr)
    logger = logging.getLogger(__name__)
    coloredlogs.install(level="DEBUG", logger=logger)
    return logger


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


def extract_sparse_sift_feature(images_list_gray):
    all_descriptors = []
    for clss in images_list_gray.keys():
        logger.debug("making sparse feature of {}".format(clss))
        for image in images_list_gray[clss]:
            descriptors = sift.detectAndCompute(image, None)
            if all_descriptors == []:
                all_descriptors = np.asarray(descriptors[1])
            else:
                all_descriptors = np.vstack(
                    (all_descriptors, np.asarray(descriptors[1])))
    return all_descriptors


def extract_dense_sift_feature(images_list_gray, step_size, name):
    logger.debug("extracting {} dense features".format(name))
    keypoints_list_dense = []
    descriptor_list_dense = []
    keypoints_list_dense = dict()
    descriptor_list_dense = dict()
    all_descriptors = []
    for clss in images_list_gray.keys():
        kp = []
        ds = []
        for image in images_list_gray[clss]:
            if image.shape == (640, 480):
                keypoint = [cv2.KeyPoint(x, y, 10) for y in range(int(image.shape[0]/(step_size[1]*2)), image.shape[0], int(image.shape[0]/step_size[1]))
                            for x in range(int(image.shape[1]/(step_size[0]*2)), image.shape[1], int(image.shape[1]/step_size[0]))]
            else:
                keypoint = [cv2.KeyPoint(x, y, 10) for y in range(int(image.shape[0]/(step_size[0]*2)), image.shape[0], int(image.shape[0]/step_size[0]))
                            for x in range(int(image.shape[1]/(step_size[1]*2)), image.shape[1], int(image.shape[1]/step_size[1]))]
            descriptor = sift.compute(image, keypoint)
            if all_descriptors == []:
                all_descriptors = np.asarray(descriptor[1])
            else:
                all_descriptors = np.vstack(
                    (all_descriptors, np.asarray(descriptor[1])))
            kp.append(keypoint)
            ds.append(descriptor)
        keypoints_list_dense[clss] = kp
        descriptor_list_dense[clss] = ds
    return all_descriptors


def calculate_histogram_sparse(mbk, dataset):
    all_hist = dict()
    for i,image_gray in dataset.items():
        dict_name = "".join(os.path.basename(i).split(".")[0].split(" "))
        logger.debug("calculating histogram {}".format(dict_name))

        image_des = sift.detectAndCompute(image_gray, None)
        prediction = mbk.predict(image_des[1])
        histogram = np.zeros(mbk.n_clusters)
        for pred in prediction:
            histogram[pred] += 1
        all_hist[dict_name] = histogram
    return all_hist


def calculate_histogram_dense(mbk, dataset, step_size=None):
    all_hist = dict()
    for i,image in dataset.items():
        dict_name = "".join(os.path.basename(i).split(".")[0].split(" "))
        logger.debug("calculating histogram {}".format(dict_name))
        if image.shape == (640, 480):
            keypoint = [cv2.KeyPoint(x, y, 10) for y in range(int(image.shape[0]/(step_size[1]*2)), image.shape[0], int(image.shape[0]/step_size[1]))
                        for x in range(int(image.shape[1]/(step_size[0]*2)), image.shape[1], int(image.shape[1]/step_size[0]))]
        else:
            keypoint = [cv2.KeyPoint(x, y, 10) for y in range(int(image.shape[0]/(step_size[0]*2)), image.shape[0], int(image.shape[0]/step_size[0]))
                        for x in range(int(image.shape[1]/(step_size[1]*2)), image.shape[1], int(image.shape[1]/step_size[1]))]
        image_des = sift.compute(image, keypoint)
        prediction = mbk.predict(image_des[1])
        histogram = np.zeros(mbk.n_clusters)
        for pred in prediction:
            histogram[pred] += 1
        all_hist[dict_name] = histogram
    return all_hist


def make_histogram(mbk, queries_image, data_image, mode="", step_size=None):
    if mode == "":
        logger.debug("no mode is selected")
        return [], []
    elif "sparse" in mode:
        logger.debug(
            "calculating histogram with images :queries , name:{}".format(mode))
        all_hist = calculate_histogram_sparse(mbk, queries_image)
        logger.debug(
            "calculating histogram with images :database , name:{}".format(mode))
        all_hist_db = calculate_histogram_sparse(mbk, data_image)
    elif "dense" in mode:
        if "8x6" in mode:
            step_size = [8, 6]
        elif "16x12" in mode:
            step_size = [16, 12]
        elif "24x18" in mode:
            step_size = [24, 18]
        elif "32x24" in mode:
            step_size = [32, 24]
        elif "64x48" in mode:
            step_size = [64, 48]
        logger.debug(
            "calculating histogram with images :database , name:{}".format(mode))
        all_hist = calculate_histogram_dense(
            mbk, queries_image, step_size=step_size)
        logger.debug(
            "calculating histogram with images :database , name:{}".format(mode))
        all_hist_db = calculate_histogram_dense(
            mbk, data_image, step_size=step_size)
    # query = cv2.cvtColor(cv2.imread("queries/QRY_BLD (00).JPG"),cv2.COLOR_BGR2GRAY)
    logger.debug("calculating power normalization")
    all_hist, all_hist_db = make_histogram_power(all_hist, all_hist_db)
    logger.debug("calculating l2 normalization")
    all_hist, all_hist_db = make_histogram_power_l2(all_hist, all_hist_db)
    return all_hist, all_hist_db


def make_histogram_power(all_hist, all_hist_db):
    all_hist_power = dict()
    all_hist_db_power = dict()
    for i in all_hist.keys():
        all_hist_power[i] = [k**0.5 for k in all_hist[i]]
        # all_hist_power.append([k**0.5 for k in i])
    for i in all_hist_db:
        all_hist_db_power[i] = [k**0.5 for k in all_hist_db[i]]
        # all_hist_db_power.append([k**0.5 for k in i])
    return all_hist_power, all_hist_db_power


def make_histogram_power_l2(all_hist_power, all_hist_db_power):
    all_hist_l2 = dict()
    all_hist_db_l2 = dict()
    for i in all_hist_power.keys():
        # all_hist_l2.append(i/np.linalg.norm(i, ord=2, axis=0))
        all_hist_l2[i] = all_hist_power[i] / \
            np.linalg.norm(all_hist_power[i], ord=2, axis=0)
    for i in all_hist_db_power.keys():
        # all_hist_db_l2.append(i/np.linalg.norm(i, ord=2, axis=0))
        all_hist_db_l2[i] = all_hist_db_power[i] / \
            np.linalg.norm(all_hist_db_power[i], ord=2, axis=0)
    return all_hist_l2, all_hist_db_l2

def train_k_means(all_descriptors, output_dir, name="sparse_feature"):
    save_path = os.path.join(output_dir, name)
    if not os.path.exists(save_path):
        logger.debug("creating directory".format(save_path))
        os.mkdir(save_path)
    mbk = dict()
    all_time = dict()
    centers = dict()
    for i in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        tmp = 0
        for j in range(0, 3):
            model_path = os.path.join(save_path, str(i)+"_"+str(j)+".pkl")
            logger.debug("clustering {} k:{} number:{}".format(name, i, j))
            mbk[str(i)+"_"+str(j)] = MiniBatchKMeans(n_clusters=i, init="random")
            t0 = time.time()
            mbk[str(i)+"_"+str(j)].fit(all_descriptors)
            logger.debug("saving cluster mbk:{}".format(model_path))
            joblib.dump(mbk[str(i)+"_"+str(j)], model_path)
            logger.debug("saved cluster mbk:{}".format(model_path))
            t_mini_batch = time.time() - t0
            all_time[str(i)+"_"+str(j)] = t_mini_batch
            time_path = model_path.split(".")[0]+"_train_time.txt"
            with open(time_path, "w") as f:
                logger.debug("writing trained time to {}".format(time_path))
                f.write("training time {}s\n".format(t_mini_batch))
            tmp += mbk[str(i)+"_"+str(j)].cluster_centers_
        centers[i] = tmp/3
    return all_time, mbk


def merge_image_list(database_dir, query_dir):
    merge_list = []
    merge_db_list = []
    for query in query_dir:
        for q in query:
            merge_list.append(q)
    for database_data in database_dir:
        for data in database_data:
            merge_db_list.append(data)
    return merge_list, merge_db_list


def query():
    pass


class_img = ["BLD", "BYC", "COW", "CTR", "FLW", "TRE"]
sift = cv2.xfeatures2d.SIFT_create()
logger = setup_logger()

all_dense_sift_name = ["dense_8x6", "dense_16x12",
                       "dense_24x18", "dense_32x24", "dense_64x48"]


def main():
    parser = argparse.ArgumentParser(
        description='A script to make clustering or use cluster to queries')
    parser.add_argument('--database_dir', '-d',
                        type=str, default="database", help="training images directory")
    parser.add_argument('--model_dir', '-m',
                        type=str, default="clusters", help="models directory")
    parser.add_argument('--queries_dir', '-q',
                        type=str, default="queries", help="queries images directory")
    parser.add_argument('--output_dir', '-o',
                        type=str, default="outputs", help="outputs of queried result directory")
    parser.add_argument("--mode_train", "-t", action="store_true", default=False,
                        help="True to train with database, False to Queries with queries_dir with database_dir")

    args = parser.parse_args()
    print(args)
    database_dir = args.database_dir
    query_dir = args.queries_dir
    model_dir = args.model_dir
    mode_train = args.mode_train

    ##### Load Images #####
    # Database
    file_list_db = read_images_list(database_dir)
    images_db, images_db_gray = read_images(file_list_db)
    # Queries
    if not mode_train:
        file_list_q = read_images_list(query_dir)
    ################################################################

    ##### Extracting Feature #####
    if mode_train:
        logger.debug("train mode")
        sparse_sift = extract_sparse_sift_feature(images_db_gray)
        all_dense_sift_name = ["dense_8x6", "dense_16x12",
                                "dense_24x18", "dense_32x24", "dense_64x48"]
        all_dense_window = [[8, 6], [16, 12], [24, 18], [32, 24], [64, 48]]
        all_dense_sift = dict()
        for index, name in enumerate(all_dense_sift_name):
            all_dense_sift[name] = extract_dense_sift_feature(
                images_db_gray, all_dense_window[index], name=name)
        ################################################################

        #### Train K-means ####
        all_time_sparse, mkbs_sparse = train_k_means(sparse_sift, model_dir)
        all_time_dense = dict()
        mkbs_dense = dict()
        for index, name in enumerate(all_dense_sift_name):
            logger.debug("training with {}".format(name))
            all_time_dense[name], mkbs_dense[name] = train_k_means(
                all_dense_sift[name], model_dir,name=name)
        exit(1)
    ################################################################
    else:
        logger.debug("queries mode")
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        logger.debug("{}".format(model_dir))
        models = sorted(glob.glob(model_dir+"/*/*.pkl"), reverse=True)
        loaded_models = dict()
        # Merge query and db file list
        logger.debug("merging file list")
        merge_list, merge_db_list = merge_image_list(
            file_list_db.values(), file_list_q.values())
        img_merge_list = dict()
        img_merge_db_list = dict()
        for img in merge_list:
            img_merge_list[img] = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)

        for img in merge_db_list:
            img_merge_db_list[img] = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)

        logger.debug("loading models")
        for model in models:
            logger.debug("loaded model {}".format(model))
            model_name = model.split("/")[1]+"_"+model.split("/")[2]
            loaded_models[model_name] = joblib.load(model)

        hists = dict()
        hists_db = dict()


        for model_index in loaded_models.keys():
            name = model_index.split(".")[0]
            logger.debug("making histogram with model {}".format(model_index))
            logger.debug("name :{}".format(name))
            all_outputs = []
            output_base_path = os.path.join(args.output_dir, name)
            if not os.path.exists(output_base_path):
                os.mkdir(output_base_path)
            else:
                logger.debug("skipping {}".format(output_base_path))
                continue
            hists[name], hists_db[name] = make_histogram(
                loaded_models[model_index], img_merge_list, img_merge_db_list, name)


            for i in hists[name]:
                logger.debug("i:{} name:{}".format(i, name))
                output_path = os.path.join(output_base_path, i+".rlt")
                tmp = []
                for j in hists_db[name]:
                    tmp.append(["{}".format(j), np.array(np.linalg.norm(
                        hists[name][i]-hists_db[name][j], ord=2, axis=0), dtype=np.float32)])
                logger.debug("writing result to {}".format(output_path))
                with open(output_path, "w") as f:
                    f.write("{}\n".format(len(hists_db[name])))
                    tmp = sorted(tmp, key=lambda x: x[1])
                    for i in tmp:
                        f.write("{} {:06f}\n".format(i[0], i[1]))

        exit(1)


if __name__ == "__main__":
    main()
