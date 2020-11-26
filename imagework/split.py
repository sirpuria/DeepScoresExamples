from PIL import Image
import os, math, random, shutil, glob
import argparse



def main(FLAGS):
    source = FLAGS.src
    destination = FLAGS.dst
    trainval_percent = FLAGS.trainval
    assert (trainval_percent>0.0 ), "trainval must be between 0 and 1"
    assert (trainval_percent<1.0), "trainval must be between 0 and 1"

    source_list = sorted(os.listdir(source))

    if not os.path.exists(os.path.join(source, "train")):
        os.makedirs(os.path.join(source, "train"))

    if not os.path.exists(os.path.join(source, "test")):
        os.makedirs(os.path.join(source, "test"))

    if not os.path.exists(os.path.join(source, "validate")):
        os.makedirs(os.path.join(source, "validate"))

    for directory in source_list:
        class_dir = os.path.join(source, directory)
        class_train_dir = os.path.join(destination, "train", directory)
        class_validate_dir = os.path.join(destination, "validate", directory)
        class_test_dir = os.path.join(destination, "test", directory)

        if not os.path.exists(class_train_dir):
            os.makedirs(class_train_dir)
        if not os.path.exists(class_test_dir):
            os.makedirs(class_test_dir)
        if not os.path.exists(class_validate_dir):
            os.makedirs(class_validate_dir)

        if(os.path.isdir(class_dir)):
            cg = class_dir + "/*.png"
            class_files = glob.glob(cg )
            random.shuffle(class_files)
            #print(class_files)
            train_num = int(len(class_files)*0.7)
            validate_num = int(len(class_files)*0.85)

            # print(train_num, "  ", len(class_files), "  ",class_dir)
            train_list = class_files[:train_num]
            validate_list = class_files[train_num:validate_num]
            test_list = class_files[validate_num:]


            for item in train_list:
                src = item
                dst = os.path.join(class_train_dir, os.path.basename(item))
                shutil.copy2(src, dst)


            for item in validate_list:
                src = item
                dst = os.path.join(class_validate_dir, os.path.basename(item))
                shutil.copy2(src, dst)

            for item in test_list:
                src = item
                dst = os.path.join(class_test_dir, os.path.basename(item))
                shutil.copy2(src, dst)







if __name__ == '__main__':
    #slice_size is the max height of the slices in pixels
    #long_slice("0.png","", os.getcwd(), 220, 120)
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--trainval', type=float)

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
