import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



# image_folder = r"phase1-task1.2-ball-detection\data\images"

# # Get list of image files
# image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]


class ImageProcess():

    def __init__(self):
        print("Processor initialized")
        
    def load_image(self,image_files,image_folder):
        # Load all images into an array
        images_array = []
        for file in image_files:
            img_path = os.path.join(image_folder, file)
            img = cv2.imread(img_path)
            if img is not None:
                images_array.append(img)
                print(f"Loaded: {file} - Shape: {img.shape}")
            else:
                print(f"Failed to load: {file}")
        return images_array


    def show_image(title, img, cmap=None):
        plt.figure(figsize=(6,6))
        if cmap:
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()

    def show_images(self,images_array,max_images,cmap= None):
        for img in images_array[:max_images]:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR→RGB for plotting

            plt.figure(figsize=(5,5))
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.show()
            

    def red_blue_mask(self,images_array,max_images):
        blue_masks = []
        red_masks = []
        for img in images_array[:max_images]:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Blue range
            lower_blue = np.array([100, 40, 40])
            upper_blue = np.array([130, 255, 255])

            # Red has 2 ranges due to hue wraparound
            lower_red1 = np.array([0, 80, 65])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 100])
            upper_red2 = np.array([180, 255, 255])

            # Create masks
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            blue_masks.append(mask_blue)
            red_masks.append(mask_red)
        return red_masks,blue_masks

    def apply_opening_morph(self,masks,max_images,kernel):
        clean_masks = []
        for mask in masks[:max_images]:
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            clean_masks.append(mask_clean)
        return clean_masks
    def detect_circles(self,mask, color_id, img_draw):
        # Apply Gaussian blur for smoother detection
        blurred = cv2.GaussianBlur(mask, (9,9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=60,
            param1=100,
            param2=29,
            minRadius=30,
            maxRadius=100
        )
        
        detections = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                detections.append((color_id, x, y, r))
                cv2.circle(img_draw, (x, y), r, (0,255,0), 2)
                cv2.circle(img_draw, (x, y), 2, (0,0,255), 3)
        return detections
    def save_all_detections(self,images_folder, detections_per_image, output_folder):
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        image_files.sort()  # ensure order matches detections list

        for img_file, detections in zip(image_files, detections_per_image):
            img_path = os.path.join(images_folder, img_file)
            h, w = cv2.imread(img_path).shape[:2]

            txt_name = os.path.splitext(img_file)[0] + ".txt"
            txt_path = os.path.join(output_folder, txt_name)

            with open(txt_path, "w") as f:
                for cls, x, y, r in detections:
                    # Normalize (x, y, width, height) to image size
                    x_center = x / w
                    y_center = y / h
                    width = (2*r) / w
                    height = (2*r) / h
                    f.write(f"{cls} {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")
# def main():
#     imgp = ImageProcess()
#     imgp.show_image(image_files)
# if __name__ == "__main__":
#     main()    