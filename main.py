import numpy as np
import matplotlib.pyplot as plt
import cv2


file_name = 'test.jpg'
plt.figure()
img = plt.imread(file_name)

def build_rows_A(point_1, point_2):
    x1,y1,_ = point_1
    x2,y2,_ = point_2
    return [[x1, y1, 1, 0, 0, 0, -x1*x2, -x2*y1, -x2],
            [0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2],]
    
def build_matrix_A(points_1, points_2):
    A = []
    for point_1, point_2 in zip(points_1, points_2):
        rows = build_rows_A(point_1, point_2)
        A += rows
    return A

def estimate_homography(points_1, points_2):
    assert len(points_1) >= 4
    assert len(points_1) >= 4
    assert len(points_1) == len(points_2)
    A = np.array(build_matrix_A(points_1, points_2))
    h_8 = np.linalg.solve(A[:, :-1], -A[:,-1])
    return np.append(h_8,1)

def reconstruct_img(H, initial_img, new_height, new_width):
    i_height, i_width, _ = initial_img.shape
    xx, yy = np.meshgrid(np.arange(0,new_height), np.arange(0,new_width))
    print(xx.shape)
    coords = np.stack([yy,xx,np.ones((new_width,new_height))])

    coords = coords.reshape((3,new_width*new_height))
    new_coords = (np.linalg.inv(H)@coords).reshape((3,new_width, new_height))
    new_coords /= new_coords[2]
    new_coords = np.round(new_coords)
    new_img = np.empty((new_height,new_width,3))
    for i in range(new_width):
        for j in range(new_height):
            x,y,_ = new_coords[:,i,j]
            new_img[j,i] = initial_img[int(y),int(x)]
    return new_img
    
pointsIm = []

def click_event(event, x, y, flags, params):
    global pointsIm
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        pointsIm.append([x,y,1])
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(img_cv2, (x,y), 20,(0,0,255),-1)
        cv2.putText(img_cv2, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow("win", img_cv2)
 

if __name__ == "__main__":
    cv2.namedWindow("win", cv2.WINDOW_NORMAL)
    # reading the image
    img_cv2 = cv2.imread(file_name, 1)
 
    # displaying the image
    cv2.imshow("win", img_cv2)
    cv2.resizeWindow("win", 1000, 1000)
 
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback("win", click_event)
 
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    
    new_H = 800
    new_W = 1500
    #[[350,650,1],[10,280,1], [750,300,1], [400,100,1]]
    H = estimate_homography(pointsIm, [[0,0,1],[0,new_H,1], [new_W,new_H,1],[new_W,0,1], ])
    H = H.reshape((3,3))
    new_img = reconstruct_img(H, img, new_H, new_W)
    print(new_img)
    if np.max(new_img) >= 2:
        plt.imshow(new_img.astype(int))
    else:
    	plt.imshow(new_img)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
