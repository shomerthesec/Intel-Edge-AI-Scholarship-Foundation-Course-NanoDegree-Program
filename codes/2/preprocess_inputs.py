import cv2
import numpy as np

def process( img , h , w ) :
    
    img= cv2 .resize(img,(w,h) )
    img= img .transpose((2,0,1))
    img= img.reshape(1 ,3, h , w )
    return img

def pose_estimation(input_image):
    
    # TODO: Preprocess the image for the pose estimation model
    # input should be [BxCxHxW]
    # resize takes ( w , h)
    # new image = [ h , w , c]
    # un effecient way 
    # pi=[preprocessed_image[2],preprocessed_image[0], preprocessed_image[1]]
    # transpose ( h , w ) gets ( w , h )
    # but transpose ( (0 , 1 , 2 ) ) changes the order of the first array inside ( (2,1,0) , (1,0,2) )
    # image = [ c , h , w] 
    preprocessed_image = np.copy(input_image)
    preprocessed_image= process(preprocessed_image,256, 456)
    
    return preprocessed_image


def text_detection(input_image):
    # TODO: Preprocess the image for the text detection model
    # input should be [BxCxHxW]
    # resize takes ( w , h)
    # new image = [ h , w , c]
    # un effecient way 
    # pi=[preprocessed_image[2],preprocessed_image[0], preprocessed_image[1]]
    # transpose ( h , w ) gets ( w , h )
    # but transpose ( (0 , 1 , 2 ) ) changes the order of the first array inside ( (2,1,0) , (1,0,2) )
    # image = [ c , h , w] 

    preprocessed_image= process(input_image, 768 , 1280)
    return preprocessed_image


def car_meta(input_image):
    # TODO: Preprocess the image for the car metadata model
    # input should be [BxCxHxW]
    # resize takes ( w , h)
    # new image = [ h , w , c]
    # un effecient way 
    # pi=[preprocessed_image[2],preprocessed_image[0], preprocessed_image[1]]
    # transpose ( h , w ) gets ( w , h )
    # but transpose ( (0 , 1 , 2 ) ) changes the order of the first array inside ( (2,1,0) , (1,0,2) )
    # image = [ c , h , w] 
    preprocessed_image=process(input_image , 72,72)
    
    return preprocessed_image


