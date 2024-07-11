import cv2
import numpy as np
import math

def create_perfect_circle(size=500, radius=200):
    image = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(image, center, radius, 255, -1, lineType=cv2.LINE_AA)
    return image

def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = float(2 * math.sqrt((math.pi) * area)) / perimeter
    return circularity

def calculate_contour_metrics(contours):
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    
    area_original = cv2.contourArea(cnt)
    perimeter_original = cv2.arcLength(cnt, True)
    circularity_original = float(2 * math.sqrt((math.pi) * area_original)) / perimeter_original
    
    hull = cv2.convexHull(cnt)
    
    area_hull = cv2.contourArea(hull)
    perimeter_hull = cv2.arcLength(hull, True)
    circularity_hull = float(2 * math.sqrt((math.pi) * area_hull)) / perimeter_hull
    
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

def process_image(img):
    # 高斯模糊
    blur_img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 二值化
    _, binary = cv2.threshold(blur_img, 128, 255, cv2.THRESH_BINARY)
    
    # 形態學操作
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilate1 = cv2.dilate(binary, kernel, iterations=2)
    erode1 = cv2.erode(dilate1, kernel, iterations=2)
    erode2 = cv2.erode(erode1, kernel, iterations=1)

    # Canny邊緣檢測
    edge = cv2.Canny(erode2, 50, 150)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 計算輪廓指標
    results = calculate_contour_metrics(contours)

    return results, img

# 主程序
if __name__ == "__main__":
    # 創建完美圓形
    perfect_circle = create_perfect_circle()

    # 計算原始白色圓的圓形度
    contours, _ = cv2.findContours(perfect_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    original_circularity = calculate_circularity(contours[0])

    print(f"Original white circle circularity: {original_circularity:.6f}")

    # 處理圖像
    results, img = process_image(perfect_circle)

    # 輸出擬合前後的面積、圓度值和比值
    if results:
        print(f"Processed original area: {results['area_original']:.2f}")
        print(f"Processed Convex Hull area: {results['area_hull']:.2f}")
        print(f"Processed Area ratio (hull/original): {results['area_ratio']:.6f}")
        print(f"Processed Original circularity: {results['circularity_original']:.6f}")
        print(f"Processed Convex Hull circularity: {results['circularity_hull']:.6f}")
        print(f"Processed Circularity ratio (hull/original): {results['circularity_ratio']:.6f}")

        # 繪製原始輪廓和凸包
        original_contour_image = np.zeros_like(img)
        hull_contour_image = np.zeros_like(img)
        
        cv2.drawContours(original_contour_image, [results['contour']], -1, (255,255,255), 1)
        cv2.drawContours(hull_contour_image, [results['hull']], -1, (255,255,255), 1)

        # 顯示結果
        cv2.imshow('Perfect Circle', perfect_circle)
        cv2.imshow('Original Contour', original_contour_image)
        cv2.imshow('Convex Hull', hull_contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No contours found.")
