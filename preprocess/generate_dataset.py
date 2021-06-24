import cv2
import numpy as np
import os
import glob


# TODO: 1. check existence of the dataset file
#    or 2. download dataset automatically
def resize2square(img, size, inter_img=cv2.INTER_CUBIC,
                  inter_mask=cv2.INTER_LINEAR):
    """
    Detect if a given argument is the source image or its mask by its channel dims
    and resize it while preserving its aspect ratio
    """
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w:
        if c is None:
            return cv2.resize(img, (size, size), inter_mask)
        else:
            return cv2.resize(img, (size, size), inter_img)
    if h > w:
        dif = h
    else:
        dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        return cv2.resize(mask, (size, size), inter_mask)
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
        return cv2.resize(mask, (size, size), inter_img)

def check_dir(dir_path):
    """
    Check if a directory exists and create it if it doesn't exist
    """
    check_folder = os.path.isdir(dir_path)

    # If folder doesn't exist, then create it.
    if not check_folder:
        os.makedirs(dir_path)
        print(f"created folder:  '{dir_path}'")
    else:
        print(f"'{dir_path}' folder already exists.")


def rgb_hist_equalize(img):
    """
    Perform histogram equalization on a RGB image
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    channels = cv2.split(ycrcb)

    cv2.equalizeHist(channels[0], channels[0])

    cv2.merge(channels, ycrcb)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)


def create_dataset_folder(input_path, output_path, image_size=224):
    """
    Create the smallleaves dataset by extracting and saving the image portions that contain
    large contours.
    """
    check_dir(output_path)

    # get masks and their corresponding img paths
    mask_paths = glob.glob(input_path + '/*seg.png')
    img_paths = list(map(lambda st: st.replace("seg", "img"), mask_paths))

    img_count = {
        'seg': 0,
        'og': 0,
        'error': 0,
        'black': 0
    }

    for i, (mask_path, img_path) in enumerate(zip(mask_paths, img_paths)):
        # read image and convert to RGB
        og_img = cv2.imread(img_path)
        og_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB)

        og_img = rgb_hist_equalize(og_img)
        og_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)

        # read mask and convert to grayscale
        mask = cv2.imread(mask_path)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # threshold and otsu
        thresh = cv2.threshold(
            gray, 1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # extract four contours with largest areas
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

        border = 10

        # loop over the contours
        for j, c in enumerate(cnts):
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            (x, y, w, h) = cv2.boundingRect(c)

            # checking for errors because some contours are too small
            try:
                roi = thresh[y - border:y + h + border,
                             x - border:x + w + border].copy()
                og_roi = og_img[y - border:y + h + border,
                                x - border:x + w + border].copy()

                # resize to square with padded black pixels
                if roi.shape == (0, 0):
                    raise ValueError

                # prevent saving all black images
                if np.sum(roi == 255) < 10:
                    img_count['black'] += 1
                    continue

                roi = resize2square(roi, image_size)
                og_roi = resize2square(og_roi, image_size)

                og_filename = "smleaf" + \
                    str(i+1).zfill(4) + "_" + str(j+1).zfill(2) + "_img.jpg"
                mask_filename = "smleaf" + \
                    str(i+1).zfill(4) + "_" + str(j+1).zfill(2) + "_seg.jpg"

                cv2.imwrite(os.path.join(output_path, og_filename), og_roi)
                img_count['og'] += 1
                cv2.imwrite(os.path.join(output_path, mask_filename), roi)
                img_count['seg'] += 1

                # ax1.imshow(og_roi)
                # ax2.imshow(roi, cmap='gray')
            except ValueError:
                img_count['error'] += 1
                print("[ValueError] img probably too small")
                continue
        print(f"saved {img_count['seg']} images")
    print(img_count)

if __name__ == "__main__":
    img_size = 224
    
    # generate train folder
    folder_path = './DenseLeaves/train'
    save_path = '../dataset/train'
    create_dataset_folder(folder_path, save_path, img_size)

    # generate train folder
    folder_path = './DenseLeaves/test'
    save_path = '../dataset/test'
    create_dataset_folder(folder_path, save_path, img_size)
