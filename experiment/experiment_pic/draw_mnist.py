# -*- coding: utf-8 -*
import xlrd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
def open_excel(file='1'):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception as e:
        print(str(e))

def excel_table_byname(file= 'mnist.xlsx',colnameindex=0,by_name=u'COLUMN5'):
     data = open_excel(file)
     table = data.sheet_by_name(by_name)
     ncols = table.ncols
     colnames = table.col_values(colnameindex)
     list = []
     for colnum in range(0, ncols):
         col = table.col_values(colnum)

         list.append(col)
     return list

def main():
    list= excel_table_byname()
    # for row in tables:
    #    print(row)
    font1 = {'size': 10}
    # line1, = plt.plot(data[0], color='cyan', linestyle='-', marker='*')
    # line2, = plt.plot(data[1], color='orange', linestyle='-', marker='.')
    # line3, = plt.plot(data[2], color='lime', linestyle='-', marker='v')
    # line4, = plt.plot(data[3], color='red', linestyle='-', marker='o')
    line1, = plt.plot(list[0], color='cyan', linestyle='-', marker='*')
    line2, = plt.plot(list[1], color='orange', linestyle='-', marker='.')
    line3, = plt.plot(list[2], color='lime', linestyle='-', marker='v')
    line4, = plt.plot(list[3], color='red', linestyle='-', marker='o')
    line5, = plt.plot(list[4], color='grey', linestyle='--', marker='')
    line6, = plt.plot(list[5], color='yellow', linestyle='--', marker='')
    line7, = plt.plot(list[6], color='fuchsia', linestyle=':', marker='')
    line8, = plt.plot(list[7], color='blue', linestyle='-.')
    line9, = plt.plot(list[8], color='green', linestyle='--')
    ll = plt.legend([line1, line2, line3, line4, line5, line6, line7, line8,line9],
                    ["MOM", "SGD", "NAG", "ISP", "PID-Kd=0.1", "PID-Kd=1", "PID-Kd=10", "PID-Kd=100","PID-Kd=155"], loc=4,
                    prop=font1, ncol=2)
    #ll = plt.legend([line1, line2, line3,line4], ["MOM", "SGD",  "NAG", "ISP"], loc=4,prop = font1)
    xminorLocator = MultipleLocator(1)
    # plt.ylim(25, 43)#c100-200
    # plt.xlim(0, 200)
    # plt.ylim(66, 77)#c10-200
    # plt.xlim(0, 200)
    plt.ylim(95.5, 99.2)#mnist)yz
    plt.xlim(0, 20)
    # plt.ylim(65, 78)#c10_yz
    # plt.xlim(0, 50)
    # plt.ylim(0, 45)#c100_yz
    # plt.xlim(0, 50)
    # plt.xticks(range(5))
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
    plt.ylabel("Valid Acc%", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.title("", fontsize=14)
    plt.show()
if __name__ =="__main__":
    main()
