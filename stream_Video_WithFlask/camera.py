
import cv2

class VideoCamera(object):
    
	def __init__(self):
		self.video = cv2.VideoCapture(0)
		self.face_case=cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
		self.eye_case=cv2.CascadeClassifier('cascade/haarcascade_eye.xml')
        # Opencvのカメラをセットします。(0)はノートパソコンならば組み込まれているカメラ
	
	def __del__(self):
		self.video.release()

	def get_frame(self):

		success, image = self.video.read()

		gary=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces=self.face_case.detectMultiScale(gary,1.3,5)

		h,w=image.shape[:2]

		fourcc=cv2.VideoWriter_fourcc(*'XVID')
		# dst=cv2.VideoWriter('outputs/test.avi',fourcc,30.0,(w,h))


		try:		
			for (x,y,w,h) in faces:
				cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0),2)
				roi_gray=gary[y:y+h,x:x+w]
				roi_color=image[y:y+h,x:x+w]
				eyes=self.eye_case.detectMultiScale(roi_gray)


			for (ex,ey,ew,eh) in eyes:
				cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)

		except Exception as e:
			print(e)

		ret, jpeg = cv2.imencode('.jpg', image)

		return jpeg.tobytes()

        # read()は、二つの値を返すので、success, imageの2つ変数で受けています。
        # OpencVはデフォルトでは raw imagesなので JPEGに変換
        # ファイルに保存する場合はimwriteを使用、メモリ上に格納したい時はimencodeを使用
        # cv2.imencode() は numpy.ndarray() を返すので .tobytes() で bytes 型に変換


if __name__ == '__main__':
	c=VideoCamera()
	c.get_frame()