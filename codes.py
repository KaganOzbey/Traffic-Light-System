import cv2
from ultralytics import YOLO
import time


model_path = "best.pt"  # Google Colab'da eğittiğim model dosyasının yolu
model = YOLO(model_path)


video_path = "traffic.mp4"  # Analiz edilecek video dosyasının yolu
cap = cv2.VideoCapture(video_path)

# Çıkış videosu için ayarlar
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Başlangıç ışık süresi
isik_suresi = 60  # Başlangıçta 60 saniye

# Son kontrol zamanı
last_check_time = time.time()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Video bittiğinde döngüyü sonlandır

    # Eğitilen modelle nesne tespiti
    results = model(frame)

    # Araç ve insan sayısını sayma
    arac_sayisi = 0
    yaya_sayisi = 0

    # Sonuçları özelleştirilmiş olarak çizme
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_id = int(class_id)  # Sınıf kimliğini tamsayıya çevirme
        label = f"{model.names[class_id]} {score:.2f}"

        # Sınıf kontrolü ve sayaç artırma
        if model.names[class_id] == "arac":
            arac_sayisi += 1
            color = (0, 0, 255)  # Araç için kırmızı
        elif model.names[class_id] == "insan":
            yaya_sayisi += 1
            color = (0, 255, 0)  # İnsan için yeşil
        else:
            color = (255, 255, 255)  # Varsayılan beyaz

        thickness = 2  # Çerçeve kalınlığı
        font_scale = 0.7  # Yazı boyutu
        font_thickness = 1  # Yazı kalınlığı

        # Çerçeve çizimi
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # Yazı ekleme
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(frame, (int(x1), int(y1) - text_height - baseline), (int(x1) + text_width, int(y1)), color, -1)
        cv2.putText(frame, label, (int(x1), int(y1) - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness + 1, lineType=cv2.LINE_AA)

    # Zamanı kontrol et ve her 5 saniyede bir ışık süresini güncelle
    current_time = time.time()
    if current_time - last_check_time >= 5:
        last_check_time = current_time  # Son kontrol zamanını güncelle

        # Işık süresi güncelleme (yaya ve araç sayısına göre)
        if arac_sayisi > yaya_sayisi:
            isik_suresi = max(5, isik_suresi * 0.95)  # Araç sayısı fazla ise ışık süresine %5 azalma, 5 saniyenin altına düşmesin
        elif yaya_sayisi > arac_sayisi:
            isik_suresi = isik_suresi * 1.05  # Yaya sayısı fazla ise ışık süresine %5 artış

    # Ekranın üst kısmında araç sayısı, insan sayısı ve ışık süresi yazdırma
    cv2.putText(frame, f"arac sayisi: {arac_sayisi}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"insan sayisi: {yaya_sayisi}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"isik suresi: {int(isik_suresi)} saniye", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # Çerçeveyi göster
    cv2.imshow("Video", frame)

    # Çıkış videosuna yaz
    out.write(frame)

    # 'q' tuşuna basıldığında döngüyü sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
out.release()
cv2.destroyAllWindows()
