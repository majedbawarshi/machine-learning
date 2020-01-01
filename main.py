
#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

dataFile = pd.read_csv('weblog.csv', sep=',')

def count_usage_per_month(data, month):

    timeArray = []

    for row in data['Time']:
        timeArray.append(row.split('/'))

    count = 0
    for elemnt in timeArray:
        if elemnt[1] == month:
            count+=1

    return count

def average_usage_per_month(col, condition):
    urlElements = (dataFile.loc[dataFile[col] == condition])

    janCount = count_usage_per_month(urlElements, 'Jan')
    febCount = count_usage_per_month(urlElements, 'Feb')
    marCount = count_usage_per_month(urlElements, 'Mar')
    aprCount = count_usage_per_month(urlElements, 'Apr')
    mayCount = count_usage_per_month(urlElements, 'May')
    junCount = count_usage_per_month(urlElements, 'Jun')
    julCount = count_usage_per_month(urlElements, 'Jul')
    augCount = count_usage_per_month(urlElements, 'Aug')
    sepCount = count_usage_per_month(urlElements, 'Sep')
    octCount = count_usage_per_month(urlElements, 'Oct')
    novCount = count_usage_per_month(urlElements, 'Nov')
    decCount = count_usage_per_month(urlElements, 'Dec')

    totalCount = janCount + febCount + marCount + aprCount + mayCount + julCount + julCount + augCount + sepCount + octCount + novCount + decCount

    janCountAvg = "{0:.2f}".format((janCount / totalCount)*100)
    febCountAvg = "{0:.2f}".format((febCount / totalCount)*100)
    marCountAvg = "{0:.2f}".format((marCount / totalCount)*100)
    aprCountAvg = "{0:.2f}".format((aprCount / totalCount)*100)
    mayCountAvg = "{0:.2f}".format((mayCount / totalCount)*100)
    junCountAvg = "{0:.2f}".format((junCount / totalCount)*100)
    julCountAvg = "{0:.2f}".format((julCount / totalCount)*100)
    augCountAvg = "{0:.2f}".format((augCount / totalCount)*100)
    sepCountAvg = "{0:.2f}".format((sepCount / totalCount)*100)
    octCountAvg = "{0:.2f}".format((octCount / totalCount)*100)
    novCountAvg = "{0:.2f}".format((novCount / totalCount)*100)
    decCountAvg = "{0:.2f}".format((decCount / totalCount)*100)

    averagePerMonth = [
        float(janCountAvg),
        float(febCountAvg),
        float(marCountAvg),
        float(aprCountAvg),
        float(mayCountAvg),
        float(julCountAvg),
        float(julCountAvg),
        float(augCountAvg),
        float(sepCountAvg),
        float(octCountAvg),
        float(novCountAvg),
        float(decCountAvg)
    ]

    return averagePerMonth


def max_in_array(array):
    count = 0
    rush_month_index = 0

    for month in avgPerMonth:
        if month > array[rush_month_index]:
            rush_month_index = count
        count += 1

    return rush_month_index

def get_month_name(index):
    switcher = {
        0:  "January",
        1:  "February",
        2:  "March",
        3:  "April",
        4:  "May",
        5:  "June",
        6:  "July",
        7:  "August",
        8:  "September",
        9:  "October",
        10: "November",
        11: "December"
    }
    return switcher[index]




print('first of all, we can see some statistics about our data')
print(dataFile.describe())

print('\nYou can see that the most used URL is:')
print(dataFile.URL.mode())

print('--------------------')

avgPerMonth = average_usage_per_month("URL", dataFile.URL.mode()[0])

print('The average of all months is: ')
print(avgPerMonth)

print('--------------------')

print(get_month_name(max_in_array(avgPerMonth)), 'is the Rush Month.')

urlElements = (dataFile.loc[dataFile.URL == dataFile.URL.mode()[0]])

months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]
janCount = count_usage_per_month(urlElements, 'Jan')
febCount = count_usage_per_month(urlElements, 'Feb')
marCount = count_usage_per_month(urlElements, 'Mar')
aprCount = count_usage_per_month(urlElements, 'Apr')
mayCount = count_usage_per_month(urlElements, 'May')
junCount = count_usage_per_month(urlElements, 'Jun')
julCount = count_usage_per_month(urlElements, 'Jul')
augCount = count_usage_per_month(urlElements, 'Aug')
sepCount = count_usage_per_month(urlElements, 'Sep')
octCount = count_usage_per_month(urlElements, 'Oct')
novCount = count_usage_per_month(urlElements, 'Nov')
decCount = count_usage_per_month(urlElements, 'Dec')

months_count = [
    janCount,
    febCount,
    marCount,
    aprCount,
    mayCount,
    junCount,
    julCount,
    augCount,
    sepCount,
    octCount,
    novCount,
    decCount
]

average = avgPerMonth

plt.scatter(months, average)
plt.xlabel("months")
plt.ylabel("average %")
plt.show()


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()


# observations
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
y = np.array(average)

# estimating coefficients
b = estimate_coef(x, y)
print("Estimated coefficients:\nb_0 = {}  \
\nb_1 = {}".format(b[0], b[1]))

# plotting regression line
plot_regression_line(x, y, b)
