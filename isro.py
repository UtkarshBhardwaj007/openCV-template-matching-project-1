import cv2
import numpy as np
import matplotlib.pyplot as plt

a = []
b = []
c = []
d = []
row = 0
column = 0
count = 0
img = cv2.imread('trial2.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('a.jpg', 0)
template2 = cv2.imread('b.jpg', 0)
template3 = cv2.imread('c.jpg', 0)

w, h = template.shape[::-1]
res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc1 = np.where( res >= threshold)  
for pt1 in zip(*loc1[::-1]): 
    cv2.rectangle(img, pt1, (pt1[0] + w, pt1[1] + h), (0,255,255), 2)
    b.append(pt1)

w2, h2 = template2.shape[::-1]
res = cv2.matchTemplate(gray,template2,cv2.TM_CCOEFF_NORMED)
threshhold = 0.8
loc2 = np.where( res >= threshold)
for pt2 in zip(*loc2[::-1]): 
    cv2.rectangle(img, pt2, (pt2[0] + w2, pt2[1] + h2), (100,255,0), 2)
    c.append(pt2)

w3, h3 = template3.shape[::-1]
res = cv2.matchTemplate(gray,template3,cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc3 = np.where( res >= threshold)  
for pt3 in zip(*loc3[::-1]): 
    cv2.rectangle(img, pt3, (pt3[0] + w3, pt3[1] + h3), (0,0,255), 2)
    d.append(pt3)

cv2.imshow('detected.jpg', img)
cv2.waitKey(0)

b.sort()
c.sort()
d.sort()

noofint = len (d)

for i in range (0, noofint):
	if d[0][0] == d[i][0]:
		row += 1

column = int (noofint/row + 1)
row = row + 1

print ("No. of rows are ", row," No. of columns are ",  column)

a = [[0]*column for r in range (row)]

height = d[1][1] - d[0][1]
width = d[row][0] - d[0][0]

for i in range (0, len(b)):
	a[int(b[i][1] / height)][int(b[i][0] / width)] = -1

for i in range (0, len(c)):
	a[int(c[i][1] / height)][int(c[i][0] / width)] = 1

dir = 'r'
def trace (l1, l2, l3, l4):
	q = [-l1,-l3]
	r = [l2,l4]
	plt.plot (r,q)

def direction (l1, l2, l3, l4, l):
	global dir
	if l == 1:
		if l4 - l2 == 0 and l3 - l1 > 0:
			dir = 'r'
		if l4 - l2 == 0 and l3 - l1 < 0:
			dir = 'l'
		if l3 - l1 == 0 and l4 - l2 > 0:
			dir = 'd'
		if l3 - l1 == 0 and l4 - l2 < 0:
			dir = 'u'
	if l == -1:
		if l4 - l2 == 0 and l3 - l1 > 0:
			dir = 'l'
		if l4 - l2 == 0 and l3 - l1 < 0:
			dir = 'r'
		if l3 - l1 == 0 and l4 - l2 > 0:
			dir = 'u'
		if l3 - l1 == 0 and l4 - l2 < 0:
			dir = 'd'
	return (dir)

def endline (l1, l2, dir):
	if l1 == row - 1 and dir == 'd':
		trace (l1, l2, l1 + 1, l2)
		print("NO")
		plt.show()
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		quit()
	if l1 == 0 and dir == 'u':
		trace (l1, l2, l1 - 1, l2)
		print("NO")
		plt.show()
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		quit()
	if l2 == 0 and dir == 'l':
		trace (l1, l2, l1, l2 - 1)
		print("NO")
		plt.show()
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		quit()
	if l2 == column - 1 and dir == 'r':
		if l1 == row - 1:
			print("YES")
		trace (l1, l2, l1, l2 + 1)
		plt.show()
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		quit()

if a[0][0] == -1:
	trace (-1, 0, 0, 0)
	trace (0, 0, 0, -1)
	plt.show()
	quit()

if a[0][0] == 1:
	dir = 'd'
	v=[0,0]
	k=[-1,0]
	plt.plot(k,v)

i = 0
j = 0
while True:
	if dir == 'd':
		if i < row - 1 and a[i + 1][j] == 0:
			trace (i, j, i + 1, j)
			endline(i + 1, j, dir)
			i += 1
			continue
		elif i < row - 1 and a[i + 1][j] == -1:
			trace (i, j, i + 1, j)
			dir = direction(i, j, i + 1, j, a[i + 1][j])
			endline(i + 1, j, dir)
			i += 1
			continue
		elif i < row - 1 and a[i + 1][j] == 1:
			trace (i, j, i + 1, j)
			dir = direction(i, j, i + 1, j, a[i + 1][j])
			endline(i + 1, j, dir)
			i += 1
			continue
	if dir == 'u':
		if i > 0 and a[i - 1][j] == 0:
			trace (i, j, i - 1, j)
			endline(i - 1, j, dir)
			i -= 1
			continue
		elif i > 0 and a[i - 1][j] == -1:
			trace (i, j, i - 1, j)
			dir = direction(i, j, i - 1, j, a[i - 1][j])
			endline(i - 1, j, dir)
			i -= 1
			continue
		elif i > 0 and a[i - 1][j] == 1:
			trace (i, j, i - 1, j)
			dir = direction(i, j, i - 1, j, a[i - 1][j])
			endline(i - 1, j, dir)
			i -= 1
			continue
	if dir == 'l':
		if j > 0 and a[i][j - 1] == 0:
			trace (i, j, i, j - 1)
			endline(i, j - 1, dir)
			j -= 1
			continue
		elif j > 0 and a[i][j - 1] == -1:
			trace (i, j, i, j - 1)
			dir = direction(i, j, i, j - 1, a[i][j - 1])
			endline(i, j - 1, dir)
			j -= 1
			continue
		elif j > 0 and a[i][j - 1] == 1:
			trace (i, j, i, j - 1)
			dir = direction(i, j, i, j - 1, a[i][j - 1])
			endline(i, j - 1, dir)
			j -= 1
			continue
	if dir == 'r':
		if j < column - 1 and a[i][j + 1] == 0:
			trace (i, j, i, j + 1)
			endline(i, j + 1, dir)
			j += 1
			continue
		elif j < column - 1 and a[i][j + 1] == -1:
			trace (i, j, i, j + 1)
			dir = direction(i, j, i, j + 1, a[i][j + 1])
			endline(i, j + 1, dir)
			j += 1
			continue
		elif j < column - 1 and a[i][j + 1] == 1:
			trace (i, j, i, j + 1)
			dir = direction(i, j, i, j + 1, a[i][j + 1])
			endline(i, j + 1, dir)
			j += 1
			continue