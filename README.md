# openCV-template-matching-project-1
This project solves the question given in the README.md file using openCV, backtracking, template matching and other necessary features.

The question is as follows:-

You are the project head of an ISRO development team working on a long-range microwave-based communication system. The communication device you are building use mirrors to deflect the microwave in certain directions. You are given an abandoned area in Pokhran to test your device. The area is divided into a grid of m_row,n_col and each cell can house only 1 mirror. The microwave starts from the cell (0,0), reflect from mirrors in different directions and has to exit from the cell (m-1,n-1).  Your task is to develop a small script that keeps tracks of all the visited cell the microwave beam goes to and return the path it took to reach (m-1,n-1 ) cell. If the beam can't reach (m-1,n-1) cell, print -1. An image is given as input which contains a grid of dimension  (m_row*n_col). Further, the grid can only have 2 possible values in them  (forward slash, backward slash). 

"forward slash" denoting a double-sided mirror tilting 45 degrees to the right 

"backward slash" denoting a double-sided mirror tilting 45 degrees to the left

m_row,n_col<100

The input will be in the form of an excel image consisting of 'forward slash' and 'backward slash' at random places.

Sample images are provided with the code.
