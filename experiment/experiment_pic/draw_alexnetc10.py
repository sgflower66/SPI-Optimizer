# -*- coding: utf-8 -*
import xlrd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import sys
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)
def open_excel(file='c100_200.xlsx'):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception as e:
        print(str(e))

def excel_table_byname(file= 'c10jf_alexnet_avg.xlsx',colnameindex=0,by_name=u'Sheet1'):
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
    # line1, = plt.plot(data[0], color='cyan', linestyle='-', marker='*')
    # line2, = plt.plot(data[1], color='orange', linestyle='-', marker='.')
    # line3, = plt.plot(data[2], color='lime', linestyle='-', marker='v')
    # line4, = plt.plot(data[3], color='red', linestyle='-', marker='o')
    line9, = plt.plot(data[8], color='cyan', linestyle='-')
    line10, = plt.plot(data[9], color='orange', linestyle='-' )
    line1, = plt.plot(data[0], color='red', linestyle='-')
    line2, = plt.plot(data[1], color='orange', linestyle='--')
    line3, = plt.plot(data[2], color='lime', linestyle=':')
    line4, = plt.plot(data[3], color='purple', linestyle='--')
    line5, = plt.plot(data[4], color='lightsteelblue', linestyle='--')
    line6, = plt.plot(data[5], color='cyan', linestyle='--')
    line7, = plt.plot(data[6], color='fuchsia', linestyle='--')
    line8, = plt.plot(data[7], color='blue', linestyle='--')


    ll = plt.legend([line9,line10,line1, line2, line3,line4,line5,line6,line7,line8,], ["MOM","SGD","SPI", "CI-β=0.01",  "CI-β=0.1", "CI-β=0.5", "CI-β=1", "CI-β=5", "CI-β=10","CI-β=1000",], loc=4,prop = font1)
#    ll = plt.legend([line1, line2, line3,line4], ["MOM", "SGD",  "NAG", "ISP"], loc=4,prop = font1)
    xminorLocator = MultipleLocator(1)
    # plt.ylim(25, 43)#c100-200
    # plt.xlim(0, 200)
    # plt.ylim(66, 77)#c10-200
    # plt.xlim(0, 200)
    # plt.ylim(97.6, 99.3)#mnist)yz
    # plt.xlim(0, 20)
    plt.ylim(65, 76.5)#c10_yz
    plt.xlim(0, 50)
    # plt.ylim(0, 45)#c100_yz
    # plt.xlim(0, 35)
    # plt.xticks(range(5))
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.0f'))
    plt.ylabel("Test Acc%", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.title("", fontsize=14)
    plt.show()
if __name__ =="__main__":
    main()
