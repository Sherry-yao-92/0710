import cv2
import numpy as np

def compute_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def adjust_sharpness(image, target_sharpness, max_iterations=10):
    current_sharpness = compute_sharpness(image)
    adjusted_image = image.copy()
    
    for _ in range(max_iterations):
        if abs(current_sharpness - target_sharpness) < 0.1:
            break
        
        # 創建銳化核心
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        
        # 應用銳化核心
        sharpened = cv2.filter2D(adjusted_image, -1, kernel)
        
        # 計算混合因子
        blend_factor = min(abs(target_sharpness - current_sharpness) / current_sharpness, 0.1)
        
        # 根據目標銳利度調整圖像
        if current_sharpness < target_sharpness:
            adjusted_image = cv2.addWeighted(adjusted_image, 1 - blend_factor, sharpened, blend_factor, 0)
        else:
            adjusted_image = cv2.addWeighted(adjusted_image, 1 + blend_factor, sharpened, -blend_factor, 0)
        
        current_sharpness = compute_sharpness(adjusted_image)
        #print(f"Current sharpness: {current_sharpness:.2f}")
    
    return adjusted_image

# 讀取圖像
img = cv2.imread('Test_images/Slight under focus/0021.tiff')

# 計算當前銳利度
current_sharpness = compute_sharpness(img)
print(f"The current sharpness of the image is: {current_sharpness:.2f}")

# 設定目標銳利度值為 70
target_sharpness = 74

# 調整銳利度
adjusted_img = adjust_sharpness(img, target_sharpness)

# 檢查調整後的銳利度
adjusted_sharpness = compute_sharpness(adjusted_img)
print(f"The final adjusted sharpness of the image is: {adjusted_sharpness:.2f}")

# 顯示原圖和調整後的圖像
cv2.imshow('Original Image', img)
cv2.imshow('Adjusted Image', adjusted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存調整後的圖像
cv2.imwrite('C:/Users/USER/HK/0710/Test_images/Slight under focus/sharpen_image.tiff', adjusted_img)