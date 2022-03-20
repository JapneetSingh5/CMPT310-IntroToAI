import numpy as np
x_coords = [-1, -1, 0, 0, 1, 1, 2, 2, 3, 3]
y_coords = [2, 1, 3, 2, 3, -1, 0, -1, 1, 0]
classification = ['+', 'o', '+', 'o', 'o', '+', '+', 'o', '+', 'o']

def euclidean_dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));

def nearest_neighbors(x_coords, y_coords, classification, x, y):
    distances = []
    for i in range(0, len(x_coords)):
        if (x!=x_coords[i]) or (y!=y_coords[i]):
            distances.append((euclidean_dist(x, y, x_coords[i], y_coords[i]), i, classification[i])) 
    distances.sort()
    return distances

def get_knn(x_coords, y_coords, classification, x, y, k):
    return nearest_neighbors(x_coords, y_coords, classification, x, y)[0: k];

def knn(x_coords, y_coords, classification, k):
    prediction = []
    for i in range(0, len(x_coords)):
        prediction.append(knn_prediction(x_coords, y_coords, classification, x_coords[i], y_coords[i], k))
    return prediction

def knn_prediction(x_coords, y_coords,classification, x, y, k):
    knn = get_knn(x_coords, y_coords, classification, x, y, k)
    PLUSes = 0
    Os = 0
    for j in range(0, k):
        if(knn[j][2]=='o'):
            Os += 1;
        else:
            PLUSes += 1
    if(Os>PLUSes):
        return 'o'
    elif(PLUSes>Os):
        return '+'
    else:
        return '='

def getLOOCVError(x_coords, y_coords, classification, k):
    wrong_prediction = 0
    for i in range(0, len(x_coords)):
        new_x_coords = x_coords[:i]+x_coords[i+1:]
        new_y_coords = y_coords[:i]+y_coords[i+1:]
        element_prediction = knn_prediction(new_x_coords, new_y_coords, classification, x_coords[i], y_coords[i], k)
        if(element_prediction != classification[i]):
            wrong_prediction += 1
    return wrong_prediction/len(x_coords)


prediction = knn(x_coords, y_coords, classification, 3)
print(prediction)
print(classification)

print(getLOOCVError(x_coords, y_coords, classification, 3))
print(getLOOCVError(x_coords, y_coords, classification, 5))
print(getLOOCVError(x_coords, y_coords, classification, 7))
print(getLOOCVError(x_coords, y_coords, classification, 9))

