# This is the project 3 of Data Mining
# This program uses selenium to download history stock data from Yahoo finace
# and perform analysis and generate graph
# stocks we use: "INTC", "MSFT", "CSCO", "AAPL", "AMZN", "GOOG", "JNPR", "VZ", "T", "S", "TMUS"

# Import the module into the program
import numpy as np
import matplotlib.pyplot as plt
import selenium.webdriver as webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys

# this function calculate the least square fit line given the x, y
# and return a tuple (m, c). y = mx + c
def lsfl(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    return (np.linalg.lstsq(A, y, rcond=None)[0])

# this function takes the pre price, predicted post price and percentage
# and return the string recommendation and percentage
def getReco(pre, post, percentage):
    check = (post - pre) / pre
    if check > percentage:
        return ("BUY", check)
    elif check < -percentage:
        return ("SELL", check)
    else:
        return ("HOLD", check)
    

# The List of stocks
stocks = ["INTC", "MSFT", "CSCO", "AAPL", "AMZN", 
          "GOOG", "JNPR", "VZ", "T", "S", "TMUS"]
stocksSize = len(stocks)

# The url for history data webpage
preUrl = "https://finance.yahoo.com/quote/"
postUrl = "/history"

# The download file location on my computer
fileUrl = "C:/Users/Yuntian/Downloads/"
# fileUrl = "C:/Users/zheng/Downloads/"

# The constant wait time for page loading
percen = 0.05 # The percentage to determain BUY, SELL or HOLD
LOAD_TIME = 10
flag = True

# Open the driver
driver = webdriver.Chrome()

# Open each webpage and start download
for key in stocks:
    # Get the url of the history page
    thisUrl = preUrl + key + postUrl
    # Go to the page
    driver.get(thisUrl)
    # Wait and find the download hypertext
    dlObj = WebDriverWait(driver, LOAD_TIME).until(
            EC.presence_of_element_located((By.LINK_TEXT, "Download Data")))
    # download the file
    driver.get(dlObj.get_attribute('href'))

# Open the file and read the data into a grand list
dataArray = None
for key in stocks:
    # Get the url of the local file
    thisUrl = fileUrl + key + '.csv'
    # Open the file
    thisFile = open(thisUrl, 'r')
    # read the file as string and split at ','
    inputStr = thisFile.read()
    fileList = inputStr.split(',')[6:-1]
    # reshape, pick up the close value and round the array for more accuracy
    closeList = np.array(fileList).reshape(-1,6)[:,4].astype(float)
    roundList = np.around(closeList, decimals = 4).reshape(-1, 1)
    # Add the list into the dataList
    if flag:
        dataArray = roundList
        flag = False
    else:
        dataArray =  np.append(dataArray, roundList, axis = 1)
    # Close the file
    thisFile.close()

# First create a normalize version of the dataArray
maxArray = np.amax(dataArray, axis = 0)
norArray = dataArray / maxArray

# Q1: using the Euclidean metric to create a dissimilarity matrix
# Record the max value and min value and position
maxval = 0
minval = 10
maxij = (0, 0)
minij = (0, 0)
disArray = np.zeros((stocksSize, stocksSize))
for i in range(stocksSize):
    for j in range(stocksSize):
        disArray[i][j] = np.linalg.norm(norArray[:,i] - norArray[:,j])
        if maxval < disArray[i][j]:
            maxij = (i, j)
            maxval = disArray[i][j]
        if minval > disArray[i][j] and disArray[i][j] != 0:
            minij = (i, j)
            minval = disArray[i][j]

# print out the result
print("Q1: Stocks with most similar behavior:")
print(stocks[minij[0]] + " and " + stocks[minij[1]])
print("with euclidean distance: " + str(minval))
print()
print("Stocks with least similar behavior:")
print(stocks[maxij[0]] + " and " + stocks[maxij[1]])
print("with euclidean distance: " + str(maxval))
print()
print()

# Plot the graph for both stock 
fig1 = plt.subplots()
min1Array = norArray[:, minij[0]]
min2Array = norArray[:, minij[1]]
plt.plot(np.arange(0,min1Array.size,1), min1Array, 'b', label = stocks[minij[0]])
plt.plot(np.arange(0,min2Array.size,1), min2Array, 'r', label = stocks[minij[1]])
plt.legend(loc='upper left')
plt.title("Most Similar Stock Plot (Normalized, Euclidean Distance)")
plt.show()

fig2 = plt.subplots()
max0Array = norArray[:, maxij[0]]
max1Array = norArray[:, maxij[1]]
plt.plot(np.arange(0,max0Array.size,1), max0Array, 'b', label = stocks[maxij[0]])
plt.plot(np.arange(0,max1Array.size,1), max1Array, 'r', label = stocks[maxij[1]])
plt.legend(loc='upper left')
plt.title("Least Similar Stock Plot (Normalized, Euclidean Distance)")
plt.show()

# Q2: use the corrcoef function to create a matrix
corrArray = np.corrcoef(norArray.T)

# Find the least correlated (closest to 0)
absCorrArr = np.abs(corrArray)
minval = absCorrArr.min()
zeroTemp = np.where(absCorrArr == minval)
minCorrIJ = (zeroTemp[0][0], zeroTemp[1][0])

# Find the most correlated (closest to 1 or -1)
# First get the mask matrix to erase all 1's in corrcoef matrix
onesArr = np.ones(absCorrArr.shape[0]).astype(int)
maskArr = abs(np.diag(onesArr) - 1)
absCorrArr = absCorrArr * maskArr
# Then find the max value in absCorrArr
maxval = absCorrArr.max()
oneTemp = np.where(absCorrArr == absCorrArr.max())
maxCorrIJ = (oneTemp[0][0], oneTemp[1][0])

# print out the result
print("Q2: Stocks with most similar behavior:")
print(stocks[maxCorrIJ[0]] + " and " + stocks[maxCorrIJ[1]])
print("with corralation coefficient: " + str(maxval))
print()
print("Stocks with least similar behavior:")
print(stocks[minCorrIJ[0]] + " and " + stocks[minCorrIJ[1]])
print("with corralation coefficient: " + str(minval))
print()
print()

# Plot the graph for both stock
fig1 = plt.subplots()
min1Array = norArray[:, minCorrIJ[0]]
min2Array = norArray[:, minCorrIJ[1]]
plt.plot(np.arange(0,min1Array.size,1), min1Array, 'b', label = stocks[minCorrIJ[0]])
plt.plot(np.arange(0,min2Array.size,1), min2Array, 'r', label = stocks[minCorrIJ[1]])
plt.legend(loc='upper left')
plt.title("Least Similar Stock Plot (Normalized, Correlation Coefficient)")
plt.show()

fig2 = plt.subplots()
max0Array = norArray[:, maxCorrIJ[0]]
max1Array = norArray[:, maxCorrIJ[1]]
plt.plot(np.arange(0,max0Array.size,1), max0Array, 'b', label = stocks[maxCorrIJ[0]])
plt.plot(np.arange(0,max1Array.size,1), max1Array, 'r', label = stocks[maxCorrIJ[1]])
plt.legend(loc='upper left')
plt.title("Most Similar Stock Plot (Normalized, Correlation Coefficient)")
plt.show()

# calculate the least square fit line for each stock and plot them
print("Q3: Least Square Fit Line and Recommendation")
xAxis = np.arange(0, dataArray.shape[0], 1)
for i in range(stocksSize):
    # get the regression line
    yAxis = dataArray[:,i]
    m, c = lsfl(xAxis, yAxis)
    # plot the graph
    fig3 = plt.subplots()
    plt.plot(xAxis, yAxis, 'b', label = stocks[i])
    plt.plot(xAxis, m * xAxis + c, 'r', label = 'Fit line')
    plt.legend(loc='upper left')
    plt.title("Stock " + stocks[i])
    plt.show()
    # predict the next day stock price
    nextprice = m * (xAxis[-1] + 1) + c
    recom, checkVal = getReco(yAxis[-1], nextprice, percen)
    print("Stock " + str(stocks[i]) + " has close price " + str(yAxis[-1]) 
             + " and predict price " + str(nextprice))
    print("Recommendation: " + str(recom) 
             + ", with percentage difference " + str(checkVal))
    print()























