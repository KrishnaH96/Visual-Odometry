  import cv2 
  import numpy as np
  from matplotlib import pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.animation as animation
  
  # Create a video capture object, in this case we are reading the video from a file
  vid_capture = cv2.VideoCapture('C:\Masters\Spring 2023\ENPM673_Perception\Perception\Project2\project2.avi')

  # camera_pose_plot = []
  x_travel_camera = []
  y_travel_camera = []
  z_travel_camera = []

  roll_plot = []
  pitch_plot = []
  yaw_plot = []

  frame_no = []

  while(vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if ret == True:

      
      scale_factor = 0.5
      resized_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

      grey_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
      # hist = cv2.calcHist([grey_image], [0], None, [256], [0, 256])

      # plt.plot(hist)
      # plt.xlim([0, 256])
      # plt.xlabel('Pixel value')
      # plt.ylabel('Frequency')
      # plt.show()
      
      mask_image = cv2.inRange(grey_image, 198, 245)
        
      mask_image= cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

      edges = cv2.Canny(mask_image, 50, 150)

      image_h, image_w = edges.shape

      step_size_acc_d = 1

      step_size_acc_theta = 1

      max_d = int(np.sqrt(image_h **2 + image_w **2))

      acc_for_hough = np.zeros((2*max_d, (180//step_size_acc_theta)))

      array_theta_d = range(0, 180, step_size_acc_theta)

      for row in range(image_h):
        for col in range(image_w):
          if edges[row][col] !=0:

            for index_theta, theta in enumerate(array_theta_d):
              acc_d = int(col * np.cos(theta * (np.pi/180)) + row * np.sin(theta * (np.pi/180)))
              acc_for_hough[acc_d + max_d][index_theta] += 1

      accu_index_with_lines = []

      for m in range(0,4):
        max_votes = 0                 
        vote_max = np.max(acc_for_hough)
        indexes_max_vote = np.argwhere(acc_for_hough == vote_max)

        for i in indexes_max_vote:
            votes = acc_for_hough[i[0], i[1]]
            if votes > max_votes:
                max_votes = votes
                update_d = i[0] - max_d
                update_theta = i[1]
        accu_index_with_lines.append((update_d,update_theta))
        acc_for_hough[indexes_max_vote[:, 0], indexes_max_vote[:, 1]] = 0

        for j in range(-10, 11):
          for k in range(-10, 11):
            d_index = update_d + j + max_d
            theta_index = update_theta+ k
            if 0 <= d_index < 2 * max_d and 0 <= theta_index < 180:
              acc_for_hough[d_index, theta_index] = 0

      for r, theta in accu_index_with_lines:
          
          # if len(accu_index_with_lines) == 4:        
              x1 = int(r*np.cos(theta* (np.pi/180)) + 1000*(-np.sin(theta*(np.pi/180))))
              y1 = int(r*np.sin(theta*(np.pi/180)) + 1000*(np.cos(theta* (np.pi/180))))

              x2 = int(r*np.cos(theta* (np.pi/180)) - 1000*(-np.sin(theta*(np.pi/180))))
              y2 = int(r*np.sin(theta*(np.pi/180)) - 1000*(np.cos(theta* (np.pi/180))))
            
              cv2.line(resized_frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
          # else:
          #    continue

      point_of_intersections = []
      
      for i in range(len(accu_index_with_lines)):
        for j in range(i+1, len(accu_index_with_lines)):
          rot_row_1, theta1 = accu_index_with_lines[i]
          rot_row_2, theta2 = accu_index_with_lines[j]

          A = np.array([[np.cos(theta1*(np.pi/180) ), np.sin(theta1*(np.pi/180))], [np.cos(theta2*(np.pi/180)), np.sin(theta2*(np.pi/180))]])
          b = np.array([rot_row_1, rot_row_2])

          intersection_pt = np.linalg.solve(A, b)
          pt = (int(intersection_pt[0]), int(intersection_pt[1]))

          if int(intersection_pt[0])>0 and int(intersection_pt[1]) > 0:
              point_of_intersections.append(pt)
                                        
          cv2.circle(resized_frame, pt, 5, (0, 0, 255), -1)

    
      if len(point_of_intersections) != 4:  
        continue     

      point_of_intersections.sort(key = lambda point: point[1], reverse=True)

      paper_points_camera = point_of_intersections

      paper_points_real = np.array(([[0,0], [21.6, 0], [21.6, 27.9], [0, 27.9]]))


      A = []
      
      for i in range(4):
          x, y = paper_points_camera[i]
          x_hat, y_hat = paper_points_real[i]
          A.append([x, y, 1, 0, 0, 0, -x_hat*x, -x_hat*y, -x_hat])
          A.append([0, 0, 0, x, y, 1, -y_hat*x, -y_hat*y, -y_hat])
          

      A = np.array(A)
      U, S, V = np.linalg.svd(A)
      homo_mat = V[-1, :].reshape((3, 3))
      homo_mat = homo_mat / homo_mat[2, 2]
    
      K = np.array([[1.38E+03 * scale_factor , 0, 9.46E+02* scale_factor],
                    [0, 1.38E+03 * scale_factor, 5.27E+02* scale_factor],
                    [0, 0, 1]])  

      K_inv = np.linalg.pinv(K)
      homo_mat_1 = homo_mat[:, 0]
      homo_mat_2 = homo_mat[:, 1]
      homo_mat_3 = homo_mat[:, 2]
      lambda_req = 1 / np.linalg.norm(np.dot(K_inv, homo_mat_1))

      rot_row_1 = lambda_req * np.dot(K_inv, homo_mat_1)
      rot_row_2 = lambda_req * np.dot(K_inv, homo_mat_2)
      rot_row_3 = np.cross(rot_row_1, rot_row_2)

      translation_mat = lambda_req * np.dot(K_inv, homo_mat_3)

      x_travel_camera.append(translation_mat[0])
      y_travel_camera.append(translation_mat[1])
      z_travel_camera.append(translation_mat[2])

      rotation_mat = np.column_stack((rot_row_1, rot_row_2, rot_row_3)) 

      P_camera = np.column_stack((np.dot(K, rotation_mat), translation_mat))

      
      inv_left_submat = np.linalg.pinv(P_camera[:, :-1])
      right_submat = P_camera[:, -1]
      camera_pos = -np.dot(inv_left_submat, right_submat)

      
      rotation_mat = P_camera[:3, :3]

      
      sin_roll = rotation_mat[2, 1]
      cos_roll = rotation_mat[2, 2]
      roll = np.arctan2(sin_roll, cos_roll)

      sin_pitch = -rotation_mat[2, 0]
      cos_pitch = np.sqrt(rotation_mat[2, 1]**2 + rotation_mat[2, 2]**2)
      pitch = np.arctan2(sin_pitch, cos_pitch)

      sin_yaw = rotation_mat[1, 0]
      cos_yaw = rotation_mat[0, 0]
      yaw = np.arctan2(sin_yaw, cos_yaw)


      roll_plot.append(roll)
      pitch_plot.append(pitch)
      yaw_plot.append(yaw)
      
        
      cv2.imshow('Frame', resized_frame)
  
      key = cv2.waitKey(1)
      
      
      if key == ord('q'):
        break
    else:
      break


  vid_capture.release()
  cv2.destroyAllWindows()


  for i in range(0,146):
    frame_no.append(i+1)


  fig = plt.figure()
  ax = fig.add_subplot(111, projection = '3d')
  ax.scatter(frame_no, x_travel_camera, y_travel_camera, z_travel_camera)
  ax.set_xlabel('X Direction')
  ax.set_ylabel('Y Direction')
  ax.set_zlabel('Z Direction')
  ax.set_title('3D plot to show the Camera Movement')

  plt.show()

  plt.plot(frame_no, roll_plot)
  plt.xlabel('Frame')
  plt.ylabel('Roll')
  plt.title('Plot of Roll change in every frame')
  plt.show()

  plt.plot(frame_no, pitch_plot)
  plt.xlabel('Frame')
  plt.ylabel('Pitch')
  plt.title('Plot of Pitch change in every frame')
  plt.show()

  plt.plot(frame_no, yaw_plot)
  plt.xlabel('Frame')
  plt.ylabel('Yaw')
  plt.title('Plot of Yaw change in every frame')
  plt.show()


  