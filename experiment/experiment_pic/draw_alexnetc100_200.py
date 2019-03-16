# -*- coding: utf-8 -*
import xlrd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
def open_excel(file='c100_200.xlsx'):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception as e:
        print(str(e))

def excel_table_byname(file= 'c100_alexnet-200.xlsx',colnameindex=0,by_name=u'Sheet1'):
     data = open_excel(file)
     table = data.sheet_by_name(by_name)
     nrows = table.nrows
     colnames = table.row_values(colnameindex)
     list = []
     for rownum in range(0, nrows):
         row = table.row_values(rownum)

         list.append(row)
     return list

def main():
    data= excel_table_byname()
    # for row in tables:
    #    print(row)
    font1 = {'size': 10}
    line1, = plt.plot(data[0], color='cyan', linestyle='-')
    line2, = plt.plot(data[1], color='orange', linestyle='-')
    line3, = plt.plot(data[2], color='lime', linestyle='-')
    line4, = plt.plot(data[3], color='red', linestyle='-')
    # line1, = plt.plot(data[0], color='red', linestyle='-',marker='o')
    # line2, = plt.plot(data[1], color='orange', linestyle='--')
    # line3, = plt.plot(data[2], color='lime', linestyle=':')
    # line4, = plt.plot(data[3], color='yellow', linestyle='--',marker='*')
    # line5, = plt.plot(data[4], color='lightsteelblue', linestyle='--',marker='v')
    # line6, = plt.plot(data[5], color='cyan', linestyle='--',marker='^')
    # line7, = plt.plot(data[6], color='fuchsia', linestyle='--',marker='|')
    # line8, = plt.plot(data[7], color='blue', linestyle='--')
    # ll = plt.legend([line1, line2, line3,line4,line5,line6,line7,line8], ["ISP", "CI-β=0.01",  "CI-β=0.1", "CI-β=0.5", "CI-β=1", "CI-β=5", "CI-β=10","CI-β=1000"], loc=4,prop = font1)
    ll = plt.legend([line1, line2, line3,line4], ["MOM", "SGD",  "NAG", "SPI"], loc=4,prop = font1)
    xminorLocator = MultipleLocator(1)
    plt.ylim(25, 43)#c100-200
    plt.xlim(0, 200)

    plt.axhline(39, ls="--", linewidth=1, color="black")
    plt.axvline(125, ls="--", linewidth=1, color="black")
    # plt.ylim(66, 77)#c10-200
    # plt.xlim(0, 200)
    # plt.ylim(97.6, 99.3)#mnist)yz
    # plt.xlim(0, 20)
    # plt.ylim(65, 78)#c10_yz
    # plt.xlim(0, 50)
    # plt.ylim(0, 45)#c100_yz
    # plt.xlim(0, 50)
    # plt.xticks(range(5))
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
    plt.ylabel("Test Acc%", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.title("", fontsize=14)
    plt.show()
if __name__ =="__main__":
    main()
