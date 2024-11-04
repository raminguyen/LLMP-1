import os
import skimage.draw
import sys
import numpy as np

sys.path.append('../')
from LLMP.util import Util

class Figure3:

  SIZE = (100, 100)


    # from codementum.org/cleveland-mcgill/
    #
    #
    # randomize data according to Cleveland84
    # specifically:
    #  - 5 numbers
    #  - add to 100
    #  - none less than 3
    #  - none greater than 39
    #  - differences greater than .1
    #
  

  def generate_datapoint():
      '''
      Generate data according to Cleveland84:
      - 5 numbers that add to 100
      - None less than 3, none greater than 39
      - Differences greater than 0.1
      '''

      def randomize_data():
          max_val = 39
          min_val = 3

          d = []
          while len(d) < 5:
              randomnumber = np.ceil(np.random.random() * max_val + min_val)
              found = False
              for i in range(len(d)):
                  if not ensure_difference(d, randomnumber):
                      found = True
                      break

              if not found:
                  d.append(randomnumber)

          return d

      def ensure_difference(A, c):
          # Ensure the difference between elements is greater than 0.1
          for i in range(len(A)):
              if abs(A[i] - c) < 0.1:
                  return False
          return True

      # Generate data and adjust to ensure the sum is exactly 100
      sum_val = -1
      while sum_val != 100:
          data = randomize_data()
          sum_val = sum(data)

          # Adjust the last element if the sum isn't exactly 100
          if sum_val != 100:
              diff = 100 - sum_val
              if 3 <= data[-1] + diff <= 39:
                  data[-1] += diff
                  sum_val = sum(data)
              else:
                  sum_val = -1  # Retry if adjustment goes out of bounds

      total_sum = sum(data)
      labels = np.array([val / total_sum for val in data], dtype=np.float32)

      # Set the largest value as 1 without affecting others
      max_label_index = np.argmax(labels)
      labels[max_label_index] = 1.0

      # Roll labels so the largest value (now exactly 1) is first
      labels = np.roll(labels, 5 - max_label_index)

      # Print the generated labels for debugging
      print("Generated Labels:", labels)

      return data, list(labels)


      #
      #
      # ATTENTION, HERE WE NEED TO ORDER THE LABELS ACCORDING
      # OUR CONVENTION
      #
      # NOW, THE MARKED ELEMENT IS AT POSITION 0 BUT ONLY IN THE
      # LABELS. THIS MEANS WE CAN NOW GO LEFT TO RIGHT (ROLLING) IN THE BARCHART
      # STARTING FROM THE MARKED ONE. AND SIMILARLY, IN THE PIE CHART WE CAN GO
      # COUNTER-CLOCKWISE STARTING FROM THE MARKED ONE.
    

  @staticmethod
  def data_to_barchart(data):
    '''
    '''
    barchart = np.zeros((100,100), dtype=bool)

    for i,d in enumerate(data):

      if i==0:
        start = 2
      else:
        start = 0
        
      left_bar = start+3+i*3+i*16
      right_bar = 3+i*3+i*16+16
      
      rr, cc = skimage.draw.line(99, left_bar, 99-int(d), left_bar)
      barchart[rr, cc] = 1
      rr, cc = skimage.draw.line(99, right_bar, 99-int(d), right_bar)
      barchart[rr, cc] = 1
      rr,cc = skimage.draw.line(99-int(d), left_bar, 99-int(d), right_bar)
      barchart[rr, cc] = 1
      
      if d == np.max(data):
        # mark the max
        barchart[90:91, left_bar+8:left_bar+9] = 1

    return barchart

  @staticmethod
  def data_to_piechart(data):
    '''
    '''
    LENGTH = 50

    piechart = np.zeros((100,100), dtype=bool)
    RADIUS = 30
    rr,cc = skimage.draw.circle_perimeter(50,50,RADIUS)
    piechart[rr,cc] = 1
    random_direction = np.random.randint(360)
    theta = -(np.pi / 180.0) * random_direction
    END = (LENGTH - RADIUS * np.cos(theta), LENGTH - RADIUS * np.sin(theta))
    rr, cc = skimage.draw.line(50, 50, int(np.round(END[0])), int(np.round(END[1])))
    piechart[rr, cc] = 1

    for i,d in enumerate(data):

      current_value = data[i]
      current_angle = (current_value / 100.) * 360.
      # print current_value, current_angle
      # print 'from', random_direction, 'to', current_angle
      theta = -(np.pi / 180.0) * (random_direction-current_angle)
      END = (50 - RADIUS * np.cos(theta), 50 - RADIUS * np.sin(theta))
      rr, cc = skimage.draw.line(50, 50, int(np.round(END[0])), int(np.round(END[1])))
      piechart[rr,cc] = 1
      
      if d == np.max(data):
        # this is the max spot
        theta = -(np.pi / 180.0) * (random_direction-current_angle/2.)
        END = (50 - RADIUS/2 * np.cos(theta), 50 - RADIUS/2 * np.sin(theta))
        rr, cc = skimage.draw.line(int(np.round(END[0])), int(np.round(END[1])), int(np.round(END[0])), int(np.round(END[1])))
        piechart[rr,cc] = 1
      
      random_direction -= current_angle

    return piechart

  @staticmethod
  def data_to_piechart_aa(data):
    '''
    '''
    piechart = np.zeros((100,100), dtype=np.float32)
    RADIUS = 30
    rr,cc,val = skimage.draw.circle_perimeter_aa(50,50,RADIUS)
    piechart[rr,cc] = val
    random_direction = np.random.randint(360)
    theta = -(np.pi / 180.0) * random_direction
    END = (50 - RADIUS * np.cos(theta), 50 - RADIUS * np.sin(theta))
    rr, cc, val = skimage.draw.line_aa(50, 50, int(np.round(END[0])), int(np.round(END[1])))
    piechart[rr, cc] = val

    for i,d in enumerate(data):

      current_value = data[i]
      current_angle = (current_value / 100.) * 360.
      # print current_value, current_angle
      # print 'from', random_direction, 'to', current_angle
      theta = -(np.pi / 180.0) * (random_direction-current_angle)
      END = (50 - RADIUS * np.cos(theta), 50 - RADIUS * np.sin(theta))
      rr, cc,val = skimage.draw.line_aa(50, 50, int(np.round(END[0])), int(np.round(END[1])))
      piechart[rr,cc] = val
      
      if d == np.max(data):
        # this is the max spot
        theta = -(np.pi / 180.0) * (random_direction-current_angle/2.)
        END = (50 - RADIUS/2 * np.cos(theta), 50 - RADIUS/2 * np.sin(theta))
        rr, cc,val = skimage.draw.line_aa(int(np.round(END[0])), int(np.round(END[1])), int(np.round(END[0])), int(np.round(END[1])))
        piechart[rr,cc] = val
      
      random_direction -= current_angle

    return piechart