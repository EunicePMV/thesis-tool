import numpy as np

def homogeneity(GLCM):
    n = GLCM.shape[0]
    hy = 0.0
    for x in range(n):
        for y in range(n):
            hy += GLCM[x, y] / (1 + (y - x) ** 2)
    return hy

def contrast(GLCM):
    n = GLCM.shape[0]
    cty = 0.0
    for x in range(n):
        for y in range(n):
            cty += GLCM[x, y] * (y - x) ** 2
    return cty

def correlation(GLCM, mean_x, mean_y, std_x, std_y):
    n = GLCM.shape[0]
    cny = 0.0
    for x in range(1, n):
        for y in range(1, n):
            cny += GLCM[x, y] * ((y - mean_y) * (x - mean_x)) / (std_y * std_x)
    return cny

def energy(GLCM):
    n = GLCM.shape[0]
    ey = np.sum(GLCM ** 2)
    return ey

# Example usage:
GLCM = np.array([[0.1, 0.2, 0.3],
                 [0.2, 0.4, 0.5],
                 [0.3, 0.5, 0.6]])

mean_x = 1.0  # Replace with the actual mean value of x
mean_y = 1.0  # Replace with the actual mean value of y
std_x = 1.0   # Replace with the actual standard deviation of x
std_y = 1.0   # Replace with the actual standard deviation of y

hy = homogeneity(GLCM)
cty = contrast(GLCM)
cny = correlation(GLCM, mean_x, mean_y, std_x, std_y)
ey = energy(GLCM)

print("Homogeneity:", hy)
print("Contrast:", cty)
print("Correlation:", cny)
print("Energy:", ey)



# RGB Mean (R, G, B): 0.37204444444444446 1.3580444444444444 0.6242148148148148
# RGB Std Dev (R, G, B): 5.041435492024758 15.064662001280993 8.122539760522077
# HSV Mean (H, S, V): 4.2857259259259255 15.126896296296296 1.4385185185185185
# LAB Mean (L, A, B): 1.1896296296296296 127.58719259259259 128.27624444444444
