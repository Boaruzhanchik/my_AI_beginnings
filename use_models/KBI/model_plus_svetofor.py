"""Рабочий код,циклы запущены через многопоточку,светофор работает,все нравится!"""
import cv2
import numpy as np
import time
import threading

age_net = cv2.dnn.readNetFromCaffe(r"C:\Dev\DIPLOM_BSTU_Kubok_molodix_inovatorov\gender-age\age_deploy.prototxt.txt",
                                   r"C:\Dev\DIPLOM_BSTU_Kubok_molodix_inovatorov\gender-age\age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe(r"C:\Dev\DIPLOM_BSTU_Kubok_molodix_inovatorov\gender-age\gender_deploy.prototxt.txt",
                                      r"C:\Dev\DIPLOM_BSTU_Kubok_molodix_inovatorov\gender-age\gender_net.caffemodel")

age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']


class FaceCaptureThread(threading.Thread):
    def __init__(self, red_duration, yellow_duration, green_duration):
        threading.Thread.__init__(self)
        self.red_duration = red_duration
        self.yellow_duration = yellow_duration
        self.green_duration = green_duration
        self.man_count = 0
        self.women_count = 0
        self.traffic_light_status = "Зеленый свет"

    def run(self):
        cascade_path = r'C:\Dev\DIPLOM_BSTU_Kubok_molodix_inovatorov\haarcascade_frontalface_default.xml'

        clf = cv2.CascadeClassifier(cascade_path)
        camera = cv2.VideoCapture(0)

        gender_dict = {0: "Man", 1: "Women"}

        start_time = time.time()
        while True:
            _, frame = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = clf.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            self.man_count = 0
            self.women_count = 0

            for (x, y, width, height) in faces:
                face_img = frame[y:y+height, x:x+width]
                blob = cv2.dnn.blobFromImage(
                    face_img, scalefactor=1.0, size=(227, 227),
                    mean=(78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False, crop=False)

                age_net.setInput(blob)
                age_preds = age_net.forward()

                age_idx = np.argmax(age_preds)

                age = age_list[age_idx]

                gender_net.setInput(blob)
                gender_preds = gender_net.forward()

                gender_idx = np.argmax(gender_preds)

                gender = gender_dict[gender_idx]

                if gender == "Man":
                    self.man_count += 1
                elif gender == "Women":
                    self.women_count += 1

                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
                cv2.putText(frame, "Age: {}".format(age),
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.putText(frame, "Gender: {}".format(gender),
                            (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            cv2.imshow('Faces', frame)

            elapsed_time = time.time() - start_time
            elapsedsed_time = time.time() - start_time
            if elapsedsed_time > 10:
                print("Кол-во {} мужчин и {} женщин".format(self.man_count, self.women_count))
                start_time = time.time()
            if elapsed_time < self.red_duration:
                self.traffic_light_status = "Красный свет"
                print("Красный свет")
            elif elapsed_time < self.red_duration + self.yellow_duration:
                self.traffic_light_status = "Желтый свет"
                print("Желтый свет")
            elif elapsed_time < self.red_duration + self.yellow_duration + self.green_duration:
                self.traffic_light_status = "Зеленый свет"
                print("Зеленый свет")
            else:
                if self.man_count + self.women_count > 10:
                    self.red_duration = 24
                    self.green_duration = 21
                else:
                    self.red_duration = 20
                    self.green_duration = 25
                self.traffic_light_status = "Красный свет"
                print("Кол-во {} мужчин и {} женщин".format(self.man_count, self.women_count))
                start_time = time.time()

            print("Текущий статус светофора: {}".format(self.traffic_light_status))

            if cv2.waitKey(1) == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

capture_thread = FaceCaptureThread(red_duration=20, yellow_duration=3, green_duration=25)
capture_thread.start()