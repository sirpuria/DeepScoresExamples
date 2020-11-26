from PIL import Image
import os, math, random, shutil, glob
import argparse



def main(FLAGS):
    source = FLAGS.src
    destination = FLAGS.dst
    num = FLAGS.num
    assert (num>0 ), "number must be between 0 and 1"
    if not os.path.exists(destination):
        os.makedirs(destination)



    source_list = os.listdir(os.path.join(source))

    for directory in source_list:
        class_dir = os.path.join(source, directory)
        dest_dir = os.path.join(destination, directory)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)



        if(os.path.isdir(class_dir)):
            cg = class_dir + "/*.png"
            class_files = glob.glob(cg )
            random.shuffle(class_files)

            if len(class_files)<num:
                rnd = len(class_files)
            else:
                rnd=num

            randlist = class_files[:rnd]


            for item in randlist:
                src = item
                dst = os.path.join(dest_dir, os.path.basename(item))
                shutil.copy2(src, dst)

        








if __name__ == '__main__':
    #slice_size is the max height of the slices in pixels
    #long_slice("0.png","", os.getcwd(), 220, 120)
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dst', type=str)
    parser.add_argument('--num', type=int)

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
