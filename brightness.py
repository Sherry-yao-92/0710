import cv2
import numpy as np

def adjust_brightness(image, target_brightness):
    # 轉換為灰度圖像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 計算當前亮度
    current_brightness = gray.mean()
    
    # 計算亮度調整因子
    brightness_factor = target_brightness / current_brightness
    
    # 將原圖像轉換為float32類型
    image_float = image.astype("float32")
    
    # 將原圖像的每個通道都乘以亮度調整因子
    adjusted_image = image_float * brightness_factor
    
    # 將調整後的圖像裁剪到0-255範圍內並轉換回uint8類型
    adjusted_image = np.clip(adjusted_image, 0, 255).astype("uint8")
    
    return adjusted_image

# 讀取圖像
img = cv2.imread('Test_images/Slight under focus/0021.tiff')

# 計算當前亮度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
current_brightness = gray.mean()
print(f"The current brightness of the image is: {current_brightness:.6f}")

# 設定目標亮度值 (例如 150)
target_brightness = 150

# 調整亮度
adjusted_img = adjust_brightness(img, target_brightness)

# 檢查調整後的亮度
adjusted_gray = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
adjusted_brightness = adjusted_gray.mean()
print(f"The adjusted brightness of the image is: {adjusted_brightness:.6f}")

# 顯示原圖和調整後的圖像
cv2.imshow('Original Image', img)
cv2.imshow('Adjusted Image', adjusted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存調整後的圖像
cv2.imwrite('C:/Users/USER/HK/0710/Test_images/Slight under focus/new_image.tiff', adjusted_img)
