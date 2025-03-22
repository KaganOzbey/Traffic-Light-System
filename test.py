import cv2
import supervision as sv
from ultralytics import YOLOv10

# YOLO modelini yükleme
model = YOLOv10('best.pt')

# Görüntü dosyasını yükleme (örneğin "image.jpg" dosyası)
image_path = "your_image4.jpg"
#your_image.jpeg
frame = cv2.imread(image_path)

# Görüntü yüklenemiyorsa hata mesajı yazdır
if frame is None:
    print(f"Görüntü dosyası açılamıyor: {image_path}")
    exit()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Modeli kullanarak tespit yapma
results = model(frame)[0]
detections = sv.Detections.from_ultralytics(results)

# Annotasyonları ekleme
annotated_image = bounding_box_annotator.annotate(
    scene=frame, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# Annotasyonlu görüntüyü gösterme
cv2.imshow("Görüntü", annotated_image)

# 'q' tuşuna basıldığında çıkış yapma
print("Görüntüyü kapatmak için bir tuşa basın...")
cv2.waitKey(0)

# Kaynakları serbest bırakma
cv2.destroyAllWindows()
