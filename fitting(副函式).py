import cv2
import numpy as np
import math
import time

def calculate_contour_metrics(contours):
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    
    # 計算原始輪廓的面積和圓度
    area_original = cv2.contourArea(cnt)
    perimeter_original = cv2.arcLength(cnt, True)
    circularity_original = float(2 * math.sqrt((math.pi) * area_original)) / perimeter_original
    
    # 計算凸包
    hull = cv2.convexHull(cnt)
    
    # 計算凸包的面積和圓度
    area_hull = cv2.contourArea(hull)
    perimeter_hull = cv2.arcLength(hull, True)
    circularity_hull = float(2 * math.sqrt((math.pi) * area_hull)) / perimeter_hull
    
    # 計算比值
    area_ratio = area_hull / area_original
    circularity_ratio = circularity_hull / circularity_original

    results = {
        "area_original": area_original,
        "area_hull": area_hull,
        "area_ratio": area_ratio,
        "circularity_original": circularity_original,
        "circularity_hull": circularity_hull,
        "circularity_ratio": circularity_ratio,
        "contour": cnt,
        "hull": hull
    }

    return results

def process_image(img_path, background_path):
    # 讀取圖片
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)

    # 記錄開始時間
    start_time = time.perf_counter()

    # 處理圖像
    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    blur_background = cv2.GaussianBlur(background, (3, 3), 0)
    substract = cv2.subtract(blur_background, blur_img)
    ret, binary = cv2.threshold(substract, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    erode1 = cv2.erode(binary, kernel)
    dilate1 = cv2.dilate(erode1, kernel)
    dilate2 = cv2.dilate(dilate1, kernel)
    erode2 = cv2.erode(dilate2, kernel)

    # 尋找輪廓
    edge = cv2.Canny(erode2, 50, 150)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 計算輪廓指標
    results = calculate_contour_metrics(contours)

    # 記錄結束時間
    end_time = time.perf_counter()

    # 計算總執行時間
    total_time = end_time - start_time

    return total_time, results, img

# 主程序
if __name__ == "__main__":
    img_path = 'Test_images/Slight under focus/sharpen_image.tiff'
    background_path = 'Test_images/Slight under focus/background.tiff'

    total_time, results, img = process_image(img_path, background_path)

    # 輸出時間統計
    print(f"Total execution time: {total_time:.6f} seconds")

    # 輸出擬合前後的面積、圓度值和比值
    if results:
        print(f"Original area: {results['area_original']:.2f}")
        print(f"Convex Hull area: {results['area_hull']:.2f}")
        print(f"Area ratio (hull/original): {results['area_ratio']:.6f}")
        print(f"Original circularity: {results['circularity_original']:.6f}")
        print(f"Convex Hull circularity: {results['circularity_hull']:.6f}")
        print(f"Circularity ratio (hull/original): {results['circularity_ratio']:.6f}")

        # 繪製原始輪廓和凸包
        original_contour_image = np.zeros_like(img)
        hull_contour_image = np.zeros_like(img)
        
        cv2.drawContours(original_contour_image, [results['contour']], -1, (255,255,255), 1)
        cv2.drawContours(hull_contour_image, [results['hull']], -1, (255,255,255), 1)

        # 顯示結果
        cv2.imshow('Original Contour', original_contour_image)
        cv2.imshow('Convex Hull', hull_contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
