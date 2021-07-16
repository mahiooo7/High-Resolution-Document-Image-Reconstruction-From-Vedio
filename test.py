import json
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

f = open('smartdoc17-dataset-sample/sample1/task_data.json')
data = json.load(f)
ref_id = data.get('reference_frame_id')
if ref_id == 0:
    ref_id = 1
dic = data.get('input_video_shape')
video_x_len = dic.get('x_len')
video_y_len = dic.get('y_len')

dic = data.get('target_image_shape')
image_x_len = dic.get('x_len')
image_y_len = dic.get('y_len')

dic = data.get('object_coord_in_ref_frame')
dic1 = dic.get('top_right')
top_right_y = dic1.get('y')
top_right_x = dic1.get('x')

dic1 = dic.get('bottom_left')
bottom_left_y = dic1.get('y')
bottom_left_x = dic1.get('x')

dic1 = dic.get('bottom_right')
bottom_right_y = dic1.get('y')
bottom_right_x = dic1.get('x')

dic1 = dic.get('top_left')
top_left_y = dic1.get('y')
top_left_x = dic1.get('x')

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	# gives indices of min-val along an axis
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

def computeSimilarity(imageA, imageB):
    return ssim(cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY),cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY))  

print(str(ref_id)+" "+str(video_x_len)+" "+str(video_y_len)+" "+str(image_x_len)+" "+str(image_y_len)+" "+str(top_right_y)+" "+str(top_right_x))
print(str(bottom_left_y)+" "+str(bottom_left_x)+" "+str(bottom_right_y)+" "+str(bottom_right_x)+" "+str(top_left_y)+" "+str(top_left_x))

pts = [(top_left_x, top_left_y), (top_right_x, top_right_y), (bottom_right_x, bottom_right_y), (bottom_left_x, bottom_left_y)]
pts = np.array(pts, dtype = "float32")

similar = 0
high = None

original = cv2.imread('smartdoc17-dataset-sample/sample1/reference_frame_01_dewarped.png')
videofile = 'smartdoc17-dataset-sample/sample1/input.mp4'
video = cv2.VideoCapture(videofile)
while(True):
    ret, frame = video.read()
    if ret == True:
        frame_no = video.get(cv2.CAP_PROP_POS_FRAMES)
        dst = four_point_transform(frame,pts)
        dst = cv2.rotate(dst,  cv2.ROTATE_90_CLOCKWISE)
        original = cv2.resize(original,(dst.shape[1],dst.shape[0]))
        score = computeSimilarity(original,dst)
        if score > similar:
            similar = score
            high = dst
        print(str(frame_no)+" "+str(similar))
        if frame_no == ref_id:
            #dst = cv2.rotate(dst,  cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite("first.png",frame)
            cv2.imwrite("second.png",dst)
            #break
    else:
        break
video.release()
cv2.destroyAllWindows()
cv2.imwrite("high.png",high)
