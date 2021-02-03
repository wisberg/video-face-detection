import cv2 as cv


def rescaleFrame(frame, scale=0.3):
    # Works for images, videos, live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


capture = cv.VideoCapture('face_video.mp4')
haar_cascade = cv.CascadeClassifier('haar_face.xml')

while True:
    isTrue, frame = capture.read()
    frame_resized = rescaleFrame(frame)

    gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)

    faces_rect = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=3)

    for(x, y, w, h) in faces_rect:
        cv.rectangle(frame_resized, (x, y), (x+w, y+h),
                     (0, 255, 0), thickness=1)

    cv.imshow('Video', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
