from PIL import Image
import os, math, random
import argparse


#https://stackoverflow.com/a/14252471
def long_slice(image_path, out_name, outdir, slice_height, slice_width, count):
    """slice an image into parts slice_size tall"""
    img = Image.open(image_path)
    width, height = img.size
    # print(width, "  ", height, "  ", image_path)
    # return
    upper = 0
    left = 0
    slices_h = int(math.ceil(height/slice_height))

    slices_w = int(math.ceil(width/slice_width))
    slices = slices_w * slices_h


    count_h = 1
    for slice_h in range(slices_h):
        #if we are at the end, set the lower bound to be the bottom of the image
        if count_h == slices_h:
            lower = height
        else:
            lower = int(count_h * slice_height)
        count_w = 1
        for slice_w in range(slices_w):
            if count_w == slices_w:
                right = width
            else:
                right = int(count_w * slice_width)


            #set the bounding box! The important bit
            bbox = (left, upper, right, lower)
            working_slice = img.crop(bbox)
            left += slice_width

            #save the slice
            working_slice.save(os.path.join(outdir,  out_name + "_" + str(count)+".png"))
            count_w +=1
            count +=1

        upper += slice_height
        left = 0
        count_h +=1

    return count

def main(FLAGS):
    source = FLAGS.source_dir
    destination = FLAGS.dest_dir
    height = FLAGS.height
    width = FLAGS.width

    source_list = os.listdir(source)

    for directory in source_list:
        class_dir = os.path.join(source, directory)
        if(os.path.isdir(class_dir)):
            class_dest_dir = os.path.join(destination, directory)
            os.makedirs(class_dest_dir)
            count =1
            for file in os.listdir(class_dir):
                if file.endswith('.png'):
                    image_file = os.path.join(class_dir, file)
                    count = long_slice(image_file, directory, class_dest_dir, height, width, count)
                    print(image_file)


if __name__ == '__main__':
    #slice_size is the max height of the slices in pixels
    #long_slice("0.png","", os.getcwd(), 220, 120)
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str)
    parser.add_argument('--dest_dir', type=str)
    parser.add_argument('--height', type=int, default=220)
    parser.add_argument('--width', type=int, default=120)
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
