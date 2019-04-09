import numpy as np
import pandas as pd
import os
import subprocess
import glob
import matplotlib.pyplot as plt
import copy
import sys
from sys import argv
from matplotlib.ticker import ScalarFormatter


class Graph1d():
    def __init__(self, *readDirPath):
        print(readDirPath)
        self.readDirTuple = readDirPath
        print("running {}/".format(argv[0]), end='')
        print("Constructer")
        print("readDirectory = '{}'".format(self.readDirTuple))

        # set defaultParaMeter
        # Para = [pandas columns, axix name]
        self.xPara = ['Points:0','x (cm)']
        self.xlim = [0, 10]
        # xlim = [0.038592, 0.074904]
        self.y_species = ['Species', 'Mass fraction']
        self.y1speciesList = ['O2','CO2','CO']
        self.y1colorList = ['r','b','g']
        # self.y1colorList = ['r','b','m','c','y','orange']
        self.y1specieslim = [0, 0.25]
        self.y2lim = ''
        self.y3lim = ''
        self.y_HRR = ['Qdot','Heat release rate (J/m^3-s)']
        self.y_T = ['T', 'T (K)']
        self.y_U = ['U:0', 'Ux (m/s)']
        self.y_p = ['p', 'Pressure (Pa)']
        self.ax1Para = self.y_species
        self.ax2Para = self.y_HRR
        self.ax3Para = self.y_T
        self.multiplePara = ''
        self.multipleValue = ''

    def setReadCsv(self, startNum, endNum, deltaNum, deltaT):
        self.startNum = startNum
        self.timeShow = 'off'
        self.legend = 'off'
        self.deltaT = deltaT * deltaNum
        self.readCsvNum = int((endNum-startNum)/deltaNum)
        print("readCsvNum = {}, deltaNum = {}\nstartTime = {}, endTime = {}".format(self.readCsvNum, deltaNum, startNum, endNum))
        self.data=[]

        # set readFile
        csvFileNum = len(os.listdir(self.readDirTuple[0]))
        print("CsvFileNum = {}".format(csvFileNum))
        for readDir in self.readDirTuple:
            data1axis = self.__readCsv(startNum, endNum, deltaNum, readDir)
            self.data.append(data1axis)

        self.paraMinMax=[]
        for i in range(len(self.readDirTuple)):
            self.paraMinMax.append(self.__detectMinToMax(self.data[i], self.y1specieslim, self.ax1Para, self.ax2Para, self.ax3Para))
        self.paraMinMax[0][0][0] = 0
        self.paraMinMax[0][1][0] = 0
        self.paraMinMax[0][2][0] = 0
        print(self.paraMinMax)
        
    def __readCsv(self, startTime, endTime, deltaNum, readCsvPath):
        axisData=[]
        readFileName = 'profile'
        self.mTocm = 100
        print("read {}/csvFile".format(readCsvPath))
        count = 0
        print(self.multiplePara)
        print(self.multipleValue)
        for i in range(startTime, endTime, deltaNum):
            axisData.append(pd.read_csv('{0}/{1}.{2}.csv'.format(readCsvPath, readFileName, i)))
            axisData[count].loc[:,self.xPara[0]] *= self.mTocm
            for para, multi in zip(self.multiplePara, self.multipleValue):
                axisData[count].loc[:, para] *= multi
            count+=1
        print("data[].columns.values=\\\n{}".format(axisData[self.readCsvNum-1].columns.values))
        print("data[[]].shape={}".format(axisData[self.readCsvNum-1].shape))
        return axisData

    def __my_makedirs(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
    
    def __detectMinToMax(self, dfList, splim, *pm):
        minList1=[]
        maxList1=[]
        minList2=[]
        maxList2=[]
        minList3=[]
        maxList3=[]
        if len(pm) == 3:
            if pm[0][0] == 'Species':
                for i in range(len(dfList)):
                    minList2.append(dfList[i].loc[:,pm[1][0]].min())
                    maxList2.append(dfList[i].loc[:,pm[1][0]].max())
                    minList3.append(dfList[i].loc[:,pm[2][0]].min())
                    maxList3.append(dfList[i].loc[:,pm[2][0]].max())
                return [splim,[min(minList2),max(maxList2)],[min(minList3),max(maxList3)]]
            else:
                for i in range(len(dfList)):
                    minList1.append(dfList[i].loc[:,pm[0][0]].min())
                    maxList1.append(dfList[i].loc[:,pm[0][0]].max())
                    minList2.append(dfList[i].loc[:,pm[1][0]].min())
                    maxList2.append(dfList[i].loc[:,pm[1][0]].max())
                    minList3.append(dfList[i].loc[:,pm[2][0]].min())
                    maxList3.append(dfList[i].loc[:,pm[2][0]].max())
                return [[min(minList1),max(maxList1)],[min(minList2),max(maxList2)],[min(minList3),max(maxList3)]]
        elif len(pm) == 2:
            if pm[0][0] == 'Species':
                for i in range(len(dfList)):
                    minList2.append(dfList[i].loc[:,pm[1][0]].min())
                    maxList2.append(dfList[i].loc[:,pm[1][0]].max())
                return [splim,[min(minList2),max(maxList2)]]
            else:
                for i in range(len(dfList)):
                    minList1.append(dfList[i].loc[:,pm[0][0]].min())
                    maxList1.append(dfList[i].loc[:,pm[0][0]].max())
                    minList2.append(dfList[i].loc[:,pm[1][0]].min())
                    maxList2.append(dfList[i].loc[:,pm[1][0]].max())
                return [[min(minList1),max(maxList1)],[min(minList2),max(maxList2)]]
        else:
            print("error -> please set more than two y axis and less than three y axis")
        
    def __make_patch_spines_invisible(self, ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def drawGraphs3yaxis(self, saveDir, imageFormat):
        print("running function \"drawGraphs\"")
        print("saveDirectory='{}'".format(saveDir))
        saveDir2 = self.ax1Para[0]+'_'+self.ax2Para[0]+'_'+self.ax3Para[0]
        self.__my_makedirs("{}/{}_{}".format(saveDir, saveDir2, imageFormat))

        self.paraMinMax[0][0] = self.y1specieslim
        if self.y2lim != '':
            self.paraMinMax[0][1] = self.y2lim
        if self.y3lim != '':
            self.paraMinMax[0][2] = self.y3lim
        print(self.paraMinMax)

        lineS = ["-", "--", "-."]
        print('dataNum={}'.format(len(self.data)))
        print(len(self.data[0]))
        for i in range(len(self.data[0])):
            fig, host = plt.subplots(nrows=1, ncols=1, figsize=(14,7))
            fig.subplots_adjust(right=1.0)
            par1 = host.twinx()
            par2 = host.twinx()
            par2.spines["right"].set_position(("axes", 1.15))
            self.__make_patch_spines_invisible(par2)
            par2.spines["right"].set_visible(True)
            p1List = []
            p2List = []
            p3List = []
            for j in range(len(self.data)):
                if self.ax1Para[0] == 'Species':
                    for spL, cL in zip(self.y1speciesList, self.y1colorList):
                        # print(spL)
                        # print(self.data[i].loc[:,self.xPara[0]])
                        # print(self.data[i].loc[:,spL])
                        p1, = host.plot(self.data[j][i].loc[:,self.xPara[0]], self.data[j][i].loc[:,spL], c=cL, alpha=0.7, linestyle=lineS[j], linewidth=3, label=spL)
                        p1List.append(p1)
                else:
                    p1, = host.plot(self.data[j][i].loc[:, self.xPara[0]], self.data[j][i].loc[:, self.ax1Para[0]], c='darkblue', alpha=0.7, linestyle=lineS[j], linewidth=3, label=self.ax1Para[0])

                p2, = par1.plot(self.data[j][i].loc[:,self.xPara[0]], self.data[j][i].loc[:,self.ax2Para[0]], c="firebrick", alpha=0.7, linestyle=lineS[j], linewidth=3, label=self.ax2Para[0])
                p2List.append(p2)

                p3, = par2.plot(self.data[j][i].loc[:,self.xPara[0]], self.data[j][i].loc[:,self.ax3Para[0]], c="black", alpha=0.7, linestyle=lineS[j], linewidth=3, label=self.ax3Para[0])
                p3List.append(p3)
                
                host.set_xlabel('{}'.format(self.xPara[1]), fontweight='bold', fontsize=24)
                host.set_ylabel('{}'.format(self.ax1Para[1]), fontweight='bold', fontsize=24)
                par1.set_ylabel('{}'.format(self.ax2Para[1]), fontweight='bold', fontsize=24)
                par2.set_ylabel('{}'.format(self.ax3Para[1]), fontweight='bold', fontsize=24)
                host.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
                host.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                #par1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.10f'))
                host.set_xlim(self.xlim[0], self.xlim[1])
                host.set_ylim(self.paraMinMax[0][0][0], self.paraMinMax[0][0][1])
                par1.set_ylim(self.paraMinMax[0][1][0], self.paraMinMax[0][1][1])
                # par1.set_ylim(top=self.paraMinMax[0][1][1])
                par2.set_ylim(self.paraMinMax[0][2][0], self.paraMinMax[0][2][1])
                host.tick_params(labelsize=24)
                par1.tick_params(labelsize=24)
                par2.tick_params(labelsize=24)
                # par1.set_yscale('log')
                # host.text(0.055, 0.23, 't = {0:.2f} sec'.format(i*deltaT), fontsize=24)
                if self.legend =='on':
                    host.legend(p1List, self.y1speciesList, loc='upper left', fontsize=20)
                    par1.legend((p2List[0], p3List[0]), (self.ax2Para[0], self.ax3Para[0]), loc='upper right', fontsize=20, bbox_to_anchor=(1,1))
                    # par2.legend(loc='upper right', fontsize=20, bbox_to_anchor=(0.948,0.905))
                    # par2.legend(fontsize=20, bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
                if self.timeShow == 'on':
                    host.text(self.xlim[0]+(self.xlim[1]-self.xlim[0])/2, self.paraMinMax[0][0][1] * 0.9, 't = {0:.2f} sec'.format((self.startNum + i)*self.deltaT), fontsize=24)
                if self.ax1Para[0] == 'Qdot':
                    host.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    host.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
                    host.yaxis.offsetText.set_fontsize(24)
                    # host.set_yscale('log')
                elif self.ax2Para[0] == 'Qdot':
                    par1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    par1.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
                    par1.yaxis.offsetText.set_fontsize(24)
                    par1.yaxis.offsetText.set_fontsize(24)
                fig.tight_layout()
                plt.savefig('{0}/{1}_{3}/graph{2:05}.{3}'.format(saveDir, saveDir2, self.startNum+i, imageFormat))
        cmd = "ffmpeg -r 30 -start_number {0} -i {1}/{2}_{3}/graph%05d.{3} -pix_fmt yuv420p {1}/{2}.mp4".format(self.startNum, saveDir, saveDir2, imageFormat)
        subprocess.call(cmd.split())
        # plt.show()

    def drawGraphs2yaxis(self, saveDir, imageFormat):
        print("running function \"drawGraphs2yaxis\"")
        print("saveDirectory='{}'".format(saveDir))
        saveDir2 = self.ax1Para[0]+'_'+self.ax2Para[0]
        self.__my_makedirs("{}/{}_{}".format(saveDir, saveDir2, imageFormat))

        self.paraMinMax[0][0] = self.y1specieslim
        if self.y2lim != '':
            self.paraMinMax[0][1] = self.y2lim
        print(self.paraMinMax)
        
        lineS = ["-", "--", "-."]
        print('dataNum={}'.format(len(self.data)))
        print(len(self.data[0]))

        for i in range(len(self.data[0])):
            fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
            ax2 = ax1.twinx()
            p1List = []
            p2List = []
            for j in range(len(self.data)):
                if self.ax1Para[0] == 'Species':
                    for spL, cL in zip(self.y1speciesList, self.y1colorList):
                        # print(spL)
                        # print(self.data[i].loc[:,self.xPara[0]])
                        # print(self.data[i].loc[:,spL])
                        spLabel = spL
                        for multiP, multiV in zip(self.multiplePara, self.multipleValue):
                            if multiP == spL:
                                spLabel = multiP + '*' + str(multiV)
                        p1, = ax1.plot(self.data[j][i].loc[:,self.xPara[0]], self.data[j][i].loc[:,spL], c=cL, alpha=0.7, linestyle=lineS[j], linewidth=5, label=spLabel)
                        p1List.append(p1)
                else:
                    p1, = ax1.plot(self.data[j][i].loc[:, self.xPara[0]], self.data[j][i].loc[:, self.ax1Para[0]], c='darkblue', alpha=0.7, linestyle=lineS[j], linewidth=5, label=self.ax1Para[0])
                p2, = ax2.plot(self.data[j][i].loc[:,self.xPara[0]], self.data[j][i].loc[:,self.ax2Para[0]], c="k", alpha=0.7, linestyle=lineS[j], linewidth=5, label=self.ax2Para[0])
                p2List.append(p2)

                ax1.set_xlabel('{}'.format(self.xPara[1]), fontweight='bold', fontsize=24)
                ax1.set_ylabel('{}'.format(self.ax1Para[1]), fontweight='bold', fontsize=24)
                ax2.set_ylabel('{}'.format(self.ax2Para[1]), fontweight='bold', fontsize=24)
                ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
                ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                # ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.10f'))
                ax1.set_xlim(self.xlim[0], self.xlim[1])
                ax1.set_ylim(self.paraMinMax[0][0][0], self.paraMinMax[0][0][1])
                ax2.set_ylim(self.paraMinMax[0][1][0], self.paraMinMax[0][1][1])
                ax1.tick_params(labelsize=24)
                ax2.tick_params(labelsize=24)
                if self.timeShow == 'on':
                    ax1.text(self.xlim[0]+(self.xlim[1]-self.xlim[0])/2, self.paraMinMax[0][0][1] * 0.9, 't = {0:.2f} sec'.format((self.startNum + i)*self.deltaT), fontsize=24)
                if self.legend =='on':
                    ax1.legend(loc='upper left', fontsize=28)
                    # ax1.legend(loc='upper left', fontsize=20, frameon=False)
                    ax2.legend(loc='upper right', fontsize=28)
                if self.ax1Para[0] == 'Qdot':
                    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    ax1.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
                    ax1.yaxis.offsetText.set_fontsize(24)
                elif self.ax2Para[0] == 'Qdot':
                    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    ax2.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
                    ax2.yaxis.offsetText.set_fontsize(24)
                fig.tight_layout()
                plt.savefig('{0}/{1}_{3}/graph{2:05}.{3}'.format(saveDir, saveDir2, self.startNum+i, imageFormat))
        cmd = "ffmpeg -r 30 -start_number {0} -i {1}/{2}_{3}/graph%05d.{3} -pix_fmt yuv420p {1}/{2}.mp4".format(self.startNum, saveDir, saveDir2, imageFormat)
        subprocess.call(cmd.split())
        # plt.show()
    
class FFmpeg():
    def __init__(self, readDirPath):
        self.readDir=readDirPath

    def imageToMov(self, readFileName, saveFileName):
        cmd = "ffmpeg -r 20 -start_number 0 -i {0}/{1} -pix_fmt yuv420p {2}".format(self.readDir, readFileName, saveFileName)
        subprocess.call(cmd.split())
    # cmd = "ffmpeg -r 20 -i {0}/result.[0334-1000].png -pix_fmt yuv420p {1}.mp4")
    # cmd = "ffmpeg -r 20 -i {0}/result.0[3-9][0-9][0-9].png -pix_fmt yuv420p {1}.mp4")
    def pngTrim(self):
        print(os.path.basename(self.readDir))
        trimDir = "{}_trim".format(os.path.basename(self.readDir))
        cmd_copy = "cp -r {0} {1}".format(self.readDir, trimDir)
        subprocess.call(cmd_copy.split())
        cmd_trim = "mogrify -trim {0}/*.png".format(trimDir)
        subprocess.call(cmd_trim.split())

class PostPeak:
    def __init__(self, readDirPath):
        self.readDir=readDirPath
        # input parameters
        self.peakNum = 2
        self.showTwFunc = 'off'
        self.showContour = 'off'
        self.mTocm=100
        self.extractMaxPara = ('Qdot')
        csvFileNum = len(os.listdir(self.readDir))
        print("csvFileNum = {}".format(csvFileNum))
        #self.startTime=607
        #self.endTime=717

    def setParameter(self, startNum, endNum, delatT, fuelName):
        self.xlim = [0, 10]
        self.startTime = startNum
        self.endTime = endNum
        self.deltaT = delatT
        print("readCsvNum = {}\nstartTime = {} endTime = {}".format(self.endTime-self.startTime, self.startTime, self.endTime))
        self.fuelName = fuelName
        if self.fuelName == 'CH4':
            self.extractParameter = ('CH4', 'O2', 'CO2', 'CO', 'H2O', 'CH2O', 'H2O2', 'OH', 'Qdot', 'T', 'U:0', 'p', 'Points:0')
        elif self.fuelName == 'nC7H16':
            self.extractParameter = ('nC7H16', 'O2', 'CO2', 'CO', 'H2O', 'CH2O', 'H2O2', 'OH', 'Qdot', 'T', 'U:0', 'p', 'Points:0')
        self.dfList = self.__readcsvAsPd()
        print("readCsvFileNum = {}".format(len(self.dfList)))

    def __readcsvAsPd(self):
        dfList=[]
        for i in range(self.startTime, self.endTime):
            df = pd.read_csv('{0}/profile.{1}.csv'.format(self.readDir, i))
            dfList.append(df.loc[:, self.extractParameter])
        return dfList

    def my_makedirs(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def TwallFunc(self, x_cm):
        Twall_min = 300
        Twall_max = 1350
        A = (Twall_max-Twall_min)/2
        B = (Twall_max+Twall_min)/2
        # coefficient = [1.1,-5.39]
        coefficient = [0.82116927, -4.20354675466]
        return A*np.tanh(float(coefficient[0])*(x_cm)+float(coefficient[1]))+B

    def __detectPeaks(self, df):
        peakLoc=[]
        peakQdot=[]
        peakT=[]
        peakTw=[]
        if self.extractMaxPara == 'Qdot':
            for i in range(len(df.loc[:, self.extractMaxPara])-2):
                if df.loc[i, self.extractMaxPara] < df.loc[i+1, self.extractMaxPara] > df.loc[i+2, self.extractMaxPara]:
                    if df.loc[i+1, self.extractMaxPara] > 1:
                        peakLoc.append(self.mTocm*df.loc[i+1, 'Points:0'])
                        peakQdot.append(df.loc[i+1, self.extractMaxPara])
                        peakT.append(df.loc[i+1, 'T'])
                        peakTw.append(self.TwallFunc(self.mTocm*df.loc[i+1, 'Points:0']))
        return [peakLoc, peakQdot, peakT, peakTw]
    
    def __makeHRRPeakList(self, sortPeakdfList):
        hrr1stPeakList=[]
        hrr2ndPeakList=[]
        hrr3rdPeakList=[]
        for	i, sortPeakdf in enumerate(sortPeakdfList):
            if sortPeakdf.shape[0] == 0:
                hrr1stPeakList.append([self.deltaT*i, None, None])
                hrr2ndPeakList.append([self.deltaT*i, None, None])
                hrr3rdPeakList.append([self.deltaT*i, None, None])
            elif sortPeakdf.shape[0] == 1:
                hrr1stPeakList.append([self.deltaT*i, sortPeakdf.at[0, 'x_cm'], sortPeakdf.at[0, 'HRR_peak'], sortPeakdf.at[0, 'T'], sortPeakdf.at[0, 'Tw']])
                hrr2ndPeakList.append([self.deltaT*i, None, None])
                hrr3rdPeakList.append([self.deltaT*i, None, None])
            elif sortPeakdf.shape[0] == 2:
                hrr1stPeakList.append([self.deltaT*i, sortPeakdf.at[0, 'x_cm'], sortPeakdf.at[0, 'HRR_peak'], sortPeakdf.at[0, 'T'], sortPeakdf.at[0, 'Tw']])
                hrr2ndPeakList.append([self.deltaT*i, sortPeakdf.at[1, 'x_cm'], sortPeakdf.at[1, 'HRR_peak'], sortPeakdf.at[0, 'T'], sortPeakdf.at[0, 'Tw']])
                hrr3rdPeakList.append([self.deltaT*i, None, None])
            else:
                hrr1stPeakList.append([self.deltaT*i, sortPeakdf.at[0, 'x_cm'], sortPeakdf.at[0, 'HRR_peak'], sortPeakdf.at[0, 'T'], sortPeakdf.at[0, 'Tw']])
                hrr2ndPeakList.append([self.deltaT*i, sortPeakdf.at[1, 'x_cm'], sortPeakdf.at[1, 'HRR_peak'], sortPeakdf.at[0, 'T'], sortPeakdf.at[0, 'Tw']])
                hrr3rdPeakList.append([self.deltaT*i, sortPeakdf.at[2, 'x_cm'], sortPeakdf.at[2, 'HRR_peak'], sortPeakdf.at[0, 'T'], sortPeakdf.at[0, 'Tw']])
        return hrr1stPeakList, hrr2ndPeakList, hrr3rdPeakList
    
    def __makeHRRPeakdf(self, sortPeakdfList):
        peak1stdf = pd.DataFrame()
        peak2nddf = pd.DataFrame()
        peak3rddf = pd.DataFrame()
        peak4thdf = pd.DataFrame()
        peak5thdf = pd.DataFrame()
        for i, sortPeakdf in enumerate(sortPeakdfList):
            if sortPeakdf.shape[0] == 0:
                pass
            else:
                sortPeakdf['t_sec'] = self.deltaT*i
                if sortPeakdf.shape[0] == 1:
                    peak1stdf[i] = sortPeakdf.loc[0]
                elif sortPeakdf.shape[0] == 2:
                    peak1stdf[i] = sortPeakdf.loc[0]
                    peak2nddf[i] = sortPeakdf.loc[1]
                elif sortPeakdf.shape[0] == 3:
                    peak1stdf[i] = sortPeakdf.loc[0]
                    peak2nddf[i] = sortPeakdf.loc[1]
                    peak3rddf[i] = sortPeakdf.loc[2]
                elif sortPeakdf.shape[0] == 4:
                    peak1stdf[i] = sortPeakdf.loc[0]
                    peak2nddf[i] = sortPeakdf.loc[1]
                    peak3rddf[i] = sortPeakdf.loc[2]
                    peak4thdf[i] = sortPeakdf.loc[3]
                else:
                    peak1stdf[i] = sortPeakdf.loc[0]
                    peak2nddf[i] = sortPeakdf.loc[1]
                    peak3rddf[i] = sortPeakdf.loc[2]
                    peak4thdf[i] = sortPeakdf.loc[3]
                    peak5thdf[i] = sortPeakdf.loc[4]
        # print(peak1stdf,'\n\n',peak2nddf,'\n\n',peak3rddf,'\n\n',peak4thdf,'\n\n',peak5thdf)
        return peak1stdf.T, peak2nddf.T, peak3rddf.T, peak4thdf.T, peak5thdf.T
    
    def detectMaximumPeak(self, dfList):
        maxLoc=[]
        maxHRR=[]
        for t in range(len(dfList)):
            # print(dfList[t].at[dfList[t]['Qdot'].idxmax(),'Points:0'])
            # print(dfList[t]['Qdot'].max())
            # print(dfList[t].at[dfList[t]['Qdot'].idxmax(),'Qdot'])
            maxLoc.append(self.mTocm*dfList[t].at[dfList[t][self.extractMaxPara].idxmax(),'Points:0'])
            maxHRR.append(dfList[t][self.extractMaxPara].max())
        return maxLoc, maxHRR
    
    ###################  save peakHRR.csv  ####################
    def __savePeakCsv(self, peakdfList, saveDir, fileName):
        os.makedirs('{}'.format(saveDir), exist_ok=True)
        f = open('{0}/{1}'.format(saveDir, fileName), 'w')
        f.write("t,")
        for i in range(6):
            for columns in peakdfList[0].columns.values:
                f.write("{}_{},".format(i, columns))
        f.write("\n")
        for i, peakdf in enumerate(peakdfList):
            f.write("{},".format(self.deltaT*i))
            for row in peakdf.itertuples(name=None):
                for cell in row[1:]:
                    f.write("{},".format(cell))
            f.write("\n")
        f.close()
    
    ###################  save maximum HRR  ####################
    def __saveMaxCsv(self, maxLoc, maxValue, saveDir, fileName):
        os.makedirs('{}'.format(saveDir), exist_ok=True)
        f = open('{0}/{1}'.format(saveDir, fileName), 'w')
        f.write("num,t,x,Tw,maxHRR\n")
        for i in range(len(maxLoc)-1):
            loc_out = maxLoc[i]
            Twall_out = self.TwallFunc(loc_out)
            maxValue_out = maxValue[i]
            f.write("{0:},{1:.5f},{2:.5f},{3:.2f},{4:.3f}\n".format(i, i*self.deltaT, loc_out, Twall_out, maxValue_out))
            f.flush()
        f.close()

    def __peakGraph_x_t(self, saveDir, *peakdf):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        if self.showContour == 'off':
            if self.peakNum == 1:
                peakdf[0].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, c='none', marker='o', alpha=1, lw=2.5, edgecolors='r', label='1st peak')
            elif self.peakNum == 2:
                peakdf[0].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, c='none', marker='o', alpha=1, lw=2.5, edgecolors='r', label='1st peak')
                peakdf[1].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, c='none', marker=',', alpha=1, lw=2.5, edgecolors='b', label='2nd peak')
            elif self.peakNum == 3:
                peakdf[0].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, c='none', marker='o', alpha=1, lw=2.5, edgecolors='r', label='1st peak')
                peakdf[1].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, c='none', marker=',', alpha=1, lw=2.5, edgecolors='b', label='2nd peak')
                peakdf[2].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, c='none', marker='^', alpha=1, lw=2.5, edgecolors='darkmagenta', label='3rd peak')
            elif self.peakNum == 4:
                peakdf[0].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, c='none', marker='o', alpha=1, lw=2.5, edgecolors='r', label='1st peak')
                peakdf[1].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, c='none', marker=',', alpha=1, lw=2.5, edgecolors='b', label='2nd peak')
                peakdf[2].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, c='none', marker='^', alpha=1, lw=2.5, edgecolors='darkmagenta', label='3rd peak')
                peakdf[3].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, c='none', marker='*', alpha=1, lw=2.5, edgecolors='darkcyan', label='4th peak')
            # ax.scatter(peakdf[0].loc[0, 'x_cm'], peakdf[0].loc[0, 't_sec'], s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='m', label='0')
            # ax.scatter(peakdf[1].loc[0, 'x_cm'], peakdf[1].loc[0, 't_sec'], s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='m', label='0')
            # ax.scatter(peakdf[0].loc[239, 'x_cm'], peakdf[0].loc[239, 't_sec'], s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='g', label='239')
            # ax.scatter(peakdf[1].loc[239, 'x_cm'], peakdf[1].loc[239, 't_sec'], s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='g', label='239')
        elif self.showContour == 'on':
            cmap = 'seismic'
            if self.peakNum == 1:
                peakdf[0].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, marker='o', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='1st')
            elif self.peakNum == 2:
                peakdf[0].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, marker='o', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='1st')
                peakdf[1].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, marker=',', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='2nd')
            elif self.peakNum == 3:
                peakdf[0].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, marker='o', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='1st')
                peakdf[1].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, marker=',', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='2nd')
                peakdf[2].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, marker='^', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='3rd')
            elif self.peakNum == 4:
                peakdf[0].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, marker='o', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='1st')
                peakdf[1].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, marker=',', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='2nd')
                peakdf[2].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, marker='^', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='3rd')
                peakdf[3].plot(kind='scatter', x='x_cm', y='t_sec', ax=ax, s=80, marker='*', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='4th')

        ax.set_xlabel('{}'.format('x (cm)'), fontweight='bold', fontsize=24)
        ax.set_ylabel('{}'.format('t (s)'), fontweight='bold', fontsize=24)
        ax.tick_params(labelsize=24)
        ax.legend(loc='upper right', fontsize=20)
        ax.set_xlim(self.xlim[0], self.xlim[1])

        # wall function
        if self.showTwFunc == 'on':
            ax2 = ax.twinx()
            ax2.plot(np.linspace(0, 10, 100), self.TwallFunc(np.linspace(0, 10, 100)), c='k', ls='-', lw=3, alpha=0.2, label='Tw')
            ax2.set_ylabel('{}'.format('Wall Temperature (K)'), FontWeight='bold', FontSize=24)
            ax2.tick_params(labelsize=24)
            ax2.legend(loc='upper right', fontsize=20)

        fig.tight_layout()
        plt.savefig('{}/x_t.png'.format(saveDir))
        plt.savefig('{}/x_t.pdf'.format(saveDir))

    def __peakGraph_x_HRR(self, saveDir, *peakdf):
        # plt.style.use('default')
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        if self.showContour == 'off':
            if self.peakNum == 1:
                peakdf[0].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='r', label='1st peak')
            elif self.peakNum == 2:
                peakdf[0].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='r', label='1st peak')
                peakdf[1].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, c='none', marker=',', alpha=1.0, lw=2.5, edgecolors='b', label='2nd peak')
            elif self.peakNum == 3:
                peakdf[0].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='r', label='1st peak')
                peakdf[1].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, c='none', marker=',', alpha=1.0, lw=2.5, edgecolors='b', label='2nd peak')
                peakdf[2].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, c='none', marker='^', alpha=1.0, lw=2.5, edgecolors='darkmagenta', label='3rd peak')
            elif self.peakNum == 4:
                peakdf[0].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='r', label='1st peak')
                peakdf[1].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, c='none', marker=',', alpha=1.0, lw=2.5, edgecolors='b', label='2nd peak')
                peakdf[2].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, c='none', marker='^', alpha=1.0, lw=2.5, edgecolors='darkmagenta', label='3rd peak')
                peakdf[3].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, c='none', marker='*', alpha=1.0, lw=2.5, edgecolors='darkcyan', label='4th peak')
            # ax.scatter(peakdf[0].loc[0, 'x_cm'], peakdf[0].loc[0, 'HRR_peak'], s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='m', label='0')
            # ax.scatter(peakdf[1].loc[0, 'x_cm'], peakdf[1].loc[0, 'HRR_peak'], s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='m', label='0')
            # ax.scatter(peakdf[0].loc[239, 'x_cm'], peakdf[0].loc[239, 'HRR_peak'], s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='g', label='239')
            # ax.scatter(peakdf[1].loc[239, 'x_cm'], peakdf[1].loc[239, 'HRR_peak'], s=80, c='none', marker='o', alpha=1.0, lw=2.5, edgecolors='g', label='239')
        elif self.showContour == 'on':
            cmap = 'seismic'
            if self.peakNum == 1:
                peakdf[0].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, marker='o', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='1st')
            elif self.peakNum == 2:
                peakdf[0].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, marker='o', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='1st')
                peakdf[1].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, marker=',', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='2nd')
            elif self.peakNum == 3:
                peakdf[0].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, marker='o', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='1st')
                peakdf[1].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, marker=',', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='2nd')
                peakdf[2].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, marker='^', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='3rd')
            elif self.peakNum == 4:
                peakdf[0].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, marker='o', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='1st')
                peakdf[1].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, marker=',', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='2nd')
                peakdf[2].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, marker='^', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='3rd')
                peakdf[3].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, marker='*', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='4th')

        ax.set_yscale('log')
        ax.set_xlabel('{}'.format('x (cm)'), fontweight='bold', fontsize=24)
        ax.set_ylabel('{}'.format('Heat release rate (J/m^3-s)'), fontweight='bold', fontsize=24)
        ax.tick_params(labelsize=24)
        ax.legend(loc='upper right', fontsize=20)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(1e+2, 5e+11)

        # wall function
        if self.showTwFunc == 'on':
            ax2 = ax.twinx()
            ax2.plot(np.linspace(0, 10, 100), self.TwallFunc(np.linspace(0, 10, 100)), c='k', ls='-', lw=3, alpha=0.2, label='Tw')
            ax2.set_ylabel('{}'.format('Wall Temperature (K)'), FontWeight='bold', FontSize=24)
            ax2.tick_params(labelsize=24)
            ax2.legend(loc='upper right', fontsize=20)

        fig.tight_layout()
        plt.savefig('{}/x_HRR.png'.format(saveDir))
        plt.savefig('{}/x_HRR.pdf'.format(saveDir))
        plt.show()

    def detectMaxValue(self, saveDir):
        # GET MAX HRR
        maxLoc, maxValue = self.detectMaximumPeak(self.dfList)
        print("maxLoc={}\n\nMaxHHR={}".format(maxLoc, maxValue))
        self.__saveMaxCsv(maxLoc, maxValue, saveDir, 'maxHRR.csv')

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        # peakdf[0].plot(kind='scatter', x='x_cm', y='HRR_peak', ax=ax, s=80, marker='o', alpha=1.0, lw=2, edgecolors='none', c='HRR_peak', cmap='{}'.format(cmap), label='1st')
        ax.plot(maxLoc, maxValue, 'o', c ='darkblue')
        ax.set_xlabel('x (cm)', FontWeight='bold', FontSize=24)
        ax.set_ylabel('Heat release rate (J/cm^3-s)', FontWeight='bold', FontSize=24)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.tick_params(labelsize=24)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(24)
        fig.tight_layout()
        plt.savefig('./{0}/maxHRR_{1}_{2}.png'.format(saveDir, self.startTime, self.endTime))
        plt.show()

    def detectPeakValue(self, saveDir):
        # GET PEAK HRR
        peakdfList=[]
        peakdfList_s=[]
        for df in self.dfList:
            peakList = self.__detectPeaks(df.loc[:,['Qdot', 'T', 'Points:0']])
            peakdfList.append(pd.DataFrame(peakList, index=['x_cm', 'HRR_peak', 'T', 'Tw']).T)
            peakdfList_s.append(pd.DataFrame(peakList, index=['x_cm', 'HRR_peak', 'T', 'Tw']).T.sort_values('HRR_peak', ascending=False).reset_index(drop=True))
        self.__savePeakCsv(peakdfList, saveDir, 'peakHRR.csv')
        self.__savePeakCsv(peakdfList_s, saveDir, 'sortPeakHRR.csv')

        # hrr1stPeakList, hrr2ndPeakList, hrr3rdPeakList = self.__makeHRRPeakList(peakdfList_s)
        # ===> hrr1stPeakList => [deltaT*i, 'x_cm', 'HRR_peak', 'T_HRRpaak', 'Tw_HRRpaak']

        # hrrPeakLoc1df, hrrPeakLoc2df, hrrPeakdLoc3f, hrrPeakLoc4df , hrrPeakLoc5df = self.__makeHRRPeakdf(peakdfList)
        hrr1stPeakdf, hrr2ndPeakdf, hrr3rdPeakdf, hrr4thPeakdf , hrr5thPeakdf = self.__makeHRRPeakdf(peakdfList_s)
        self.__peakGraph_x_t(saveDir, hrr1stPeakdf, hrr2ndPeakdf, hrr3rdPeakdf, hrr4thPeakdf , hrr5thPeakdf)
        self.__peakGraph_x_HRR(saveDir, hrr1stPeakdf, hrr2ndPeakdf, hrr3rdPeakdf, hrr4thPeakdf , hrr5thPeakdf)

    
class PropagationSpeed(PostPeak):
    def __init__(self, readDir):
        super(PropagationSpeed, self).__init__(readDir)
        self.saveDir = None
        self.maxLoc = None
        self.maxValue = None
        self.xlim = [0, 10]
        self.ylim = [100, -500]

    def calcPropagatingSpeed(self):
        self.my_makedirs(self.saveDir)
        maxLoc, maxValue = self.detectMaximumPeak(self.dfList)
        print(len(maxLoc))
        print(len(maxValue))
        fs = []  # flame speed
        for i in range(len(maxLoc)-1):
            fs.append((maxLoc[i+1] - maxLoc[i]) / self.deltaT)
        print(fs)
        print(len(fs))

        #####   plot   ####
        maxLoc.pop()
        maxValue.pop()
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        ax.plot(maxLoc, fs, 'o', color='darkslateblue')
        ax.set_xlabel('x (cm)', FontWeight='bold', FontSize=24)
        ax.set_ylabel('Progation speed (cm/s)', FontWeight='bold', FontSize=24)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.tick_params(labelsize=24)
        # ax.set_title('FREI', FontWeight='bold')
        fig.tight_layout()
        plt.savefig('{}/propagationSpeed.png'.format(self.saveDir))

        fig, ax_ = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        ax_.plot(maxLoc, maxValue, 'o', color='darkred')
        ax_.set_xlabel('x (cm)', FontWeight='bold', FontSize=24)
        ax_.set_ylabel('Heat release rate (J/cm^3-s)', FontWeight='bold', FontSize=24)
        ax_.set_xlim(self.xlim[0], self.xlim[1])
        ax_.tick_params(labelsize=24)
        ax_.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax_.yaxis.offsetText.set_fontsize(24)
        fig.tight_layout()
        plt.savefig('{}/maxHRR.png'.format(self.saveDir))
        plt.show()

    def calcIgEx(self):
        maxLoc, maxValue = self.detectMaximumPeak(self.dfList)
        maxAll = max(maxValue)
        print(maxAll)
        quenchingPara = 0.001
        for i in range(len(maxValue)):
            if maxValue[i] < quenchingPara*maxAll:
                print(maxValue[i])
                maxValue[i] = 0
        print(maxValue)

        ######### detect ignition and extinction point ##########
        ignitionNum = []
        ignitionTime = []
        ignitionPosition = []
        extinctionNum = []
        extinctionTime = []
        extinctionPosition = []
        for i in range(len(maxValue)-1):
            if maxValue[i] == 0 and maxValue[i+1] != 0:
                ignitionNum.append(i+1)
                ignitionTime.append((i+1)*self.deltaT)
                ignitionPosition.append(maxLoc[i+1])
            if maxValue[i] != 0 and maxValue[i+1] == 0:
                extinctionNum.append(i+1)
                extinctionTime.append((i+1)*self.deltaT)
                extinctionPosition.append(maxLoc[i])
        #print(ignitionPosition)
        #for num in ignitionNum:
        #	print(maxLoc[num])


        ############### obtain cycle of FREI ##################
        numIntervalIgnition= []
        numIntervalExtinction= []
        timeIntervalIgnition= []
        timeIntervalExtinction= []
        for i in range(len(ignitionNum)-1):
            numIntervalIgnition.append(ignitionNum[i+1]-ignitionNum[i])
            timeIntervalIgnition.append(ignitionTime[i+1]-ignitionTime[i])
        for i in range(len(extinctionNum)-1):
            numIntervalExtinction.append(extinctionNum[i+1]-extinctionNum[i])
            timeIntervalExtinction.append(extinctionTime[i+1]-extinctionTime[i])

        ############  obtain time and distance from ignition to extinction  ############
        flamePropagetingTime=[]
        flamePropagetingDistance=[]
        flameQuenchingTime=[]
        flameQuenchingDistance=[]
        if ignitionNum[0] > extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(len(ignitionNum)-1):
                    flamePropagetingTime.append(extinctionTime[i+1] - ignitionTime[i])
                    flamePropagetingDistance.append(ignitionPosition[i] - extinctionPosition[i+1])
                for i in range(len(ignitionNum)):
                    flameQuenchingTime.append(ignitionTime[i] - extinctionTime[i])
                    flameQuenchingDistance.append(ignitionPosition[i] - extinctionPosition[i])
            else:
                for i in range(len(ignitionNum)):
                    flamePropagetingTime.append(extinctionTime[i+1] - ignitionTime[i])
                    flamePropagetingDistance.append(ignitionPosition[i] - extinctionPosition[i+1])
                    flameQuenchingTime.append(ignitionTime[i] - extinctionTime[i])
                    flameQuenchingDistance.append(ignitionPosition[i] - extinctionPosition[i])
        elif ignitionNum[0] < extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(len(extinctionNum)):
                    flamePropagetingTime.append(extinctionTime[i] - ignitionTime[i])
                    flamePropagetingDistance.append(ignitionPosition[i] - extinctionPosition[i])
                for i in range(len(extinctionNum)-1):
                    flameQuenchingTime.append(ignitionTime[i+1] - extinctionTime[i])
                    flameQuenchingDistance.append(ignitionPosition[i+1] - extinctionPosition[i])
            else:
                for i in range(len(extinctionNum)):
                    flamePropagetingTime.append(extinctionTime[i] - ignitionTime[i])
                    flamePropagetingDistance.append(ignitionPosition[i] - extinctionPosition[i])
                    flameQuenchingTime.append(ignitionTime[i+1] - extinctionTime[i])
                    flameQuenchingDistance.append(ignitionPosition[i+1] - extinctionPosition[i])

        """
        ############  obtain average propagation speed  ############
        averagePropagationSpeed = []
        for i in range(len(flamePropagetingTime)):
            averagePropagationSpeed.append(-flamePropagetingDistance[i]/flamePropagetingTime[i])

        #########  obtain average convection speed when quenching  #########
        averageConvectionSpeed = []
        for i in range(len(flameQuenchingTime)):
            averageConvectionSpeed.append(flameQuenchingDistance[i]/flameQuenchingTime[i])

        ###################  save results2  ####################
        f2 = open('./{}/result2.csv'.format(self.saveDir),'w')
        f2.write("num,igTime,igLoc,igTw,Tig,exTime,exLoc,exTw,Tex,ig_exTime,ig_exDist,Vave,ex_igTime,ex_igDist,Vconv\n")
        if len(timeIntervalIgnition) < len(timeIntervalExtinction):
            loopNum = len(timeIntervalIgnition)
        else:
            loopNum = len(timeIntervalExtinction)
        for i in range(loopNum):
            out1=ignitionTime[i]
            out2=ignitionPosition[i]
            out3=self.TwallFunc(out2)
            out4=timeIntervalIgnition[i]
            out5=extinctionTime[i]
            out6=extinctionPosition[i]
            out7=self.TwallFunc(out6)
            out8=timeIntervalExtinction[i]
            out9=flamePropagetingTime[i]
            out10=flamePropagetingDistance[i]
            out11=averagePropagationSpeed[i]
            out12=flameQuenchingTime[i]
            out13=flameQuenchingDistance[i]
            out14=averageConvectionSpeed[i]
            f2.write("{0:},{1:.5f},{2:.3f},{3:.2f},{4:.6f},{5:.5f},{6:.3f},{7:.2f},{8:.6f},{9:.5f},{10:.4f},{11:.3f},{12:.5f},{13:.3f},{14:.3f}\n"\
                    .format(i,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11,out12,out13,out14))
            f2.flush()
        f2.close()
        """

        ###########  delete list while quenching  ############
        t=[]
        [t.append(i*self.deltaT) for i in range(len(maxLoc))]
        delNum = []
        delNum.extend(list(range(ignitionNum[0])))
        if ignitionNum[0] > extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(1,len(ignitionNum)):
                    delNum.extend(list(range(extinctionNum[i],ignitionNum[i])))
            else:
                for i in range(1,len(ignitionNum)):
                    delNum.extend(list(range(extinctionNum[i],ignitionNum[i])))
                delNum.extend(list(range(extinctionNum[len(ignitionNum)-1],len(maxValue))))
        elif ignitionNum[0] < extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(len(extinctionNum)-1):
                    delNum.extend(list(range(extinctionNum[i],ignitionNum[i+1])))
                delNum.extend(list(range(extinctionNum[len(extinctionNum)-1],len(maxValue))))
            else:
                for i in range(len(extinctionNum)):
                    delNum.extend(list(range(extinctionNum[i],ignitionNum[i+1])))
        dellist = lambda items, indexes: [item for index, item in enumerate(items) if index not in indexes]
        t_propagating = dellist(t, delNum)
        maxLoc_propagating = dellist(maxLoc, delNum)
        maxValue_propagating = dellist(maxValue, delNum)

        ############  obtain propagation speed  ############
        propagationSpeed = []
        threshold_ignition_extinction = 1.0
        for i in range(len(maxLoc_propagating)-1):
            propagationSpeed.append((maxLoc_propagating[i+1]-maxLoc_propagating[i])/self.deltaT)

        ###################  save results3  ####################
        f3 = open('./{}/result3.csv'.format(self.saveDir),'w')
        f3.write("num,t,x,Tw,hrr,v\n")
        for i in range(len(propagationSpeed)-1):
            t_out=t_propagating[i]
            x_out=maxLoc_propagating[i]
            Twall_out=self.TwallFunc(x_out)
            max_out=maxValue_propagating[i]
            speed_out=propagationSpeed[i]
            f3.write("{0:},{1:.4f},{2:.3f},{3:.2f},{4:.1f},{4:}\n".format(i, t_out, x_out, Twall_out, max_out, speed_out))
            f3.flush()
        f3.close()

        ###########  obtain list while propagating  ############
        xEachCycle=[]
        maxValueEachCycle=[]
        if ignitionNum[0] > extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(0,len(ignitionNum)-1):
                    xEachCycle.append(maxLoc[ignitionNum[i]:extinctionNum[i+1]])
                    maxValueEachCycle.append(list(maxValue[ignitionNum[i]:extinctionNum[i+1]]))
            else:
                for i in range(0,len(ignitionNum)):
                    xEachCycle.append(maxLoc[ignitionNum[i]:extinctionNum[i+1]])
                    maxValueEachCycle.append(list(maxValue[ignitionNum[i]:extinctionNum[i+1]]))
        elif ignitionNum[0] < extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(len(extinctionNum)):
                    xEachCycle.append(maxLoc[ignitionNum[i]:extinctionNum[i]])
                    maxValueEachCycle.append(list(maxValue[ignitionNum[i]:extinctionNum[i]]))
            else:
                for i in range(len(extinctionNum)):
                # for i in range(len(extinctionNum)+1):  #from ignition to ... before extinction
                    xEachCycle.append(maxLoc[ignitionNum[i]:extinctionNum[i]])
                    maxValueEachCycle.append(list(maxValue[ignitionNum[i]:extinctionNum[i]]))

        ############  obtain propagation speed each cycle  ############
        x_cm_eachCycle_speed = []
        max_eachCycle_speed = []
        propagationSpeed_eachCycle = []
        temp_x = []
        temp_maxValue = []
        temp_propagationSpeed = []
        for i in range(len(xEachCycle)):
            temp_x.clear()
            temp_maxValue.clear()
            temp_propagationSpeed.clear()
            for j in range(len(xEachCycle[i])-1):
                temp_x.append(xEachCycle[i][j])
                temp_maxValue.append(maxValueEachCycle[i][j])
                temp_propagationSpeed.append((xEachCycle[i][j+1]-xEachCycle[i][j])/self.deltaT)
            x_cm_eachCycle_speed.append(copy.deepcopy(temp_x))
            max_eachCycle_speed.append(copy.deepcopy(temp_maxValue))
            propagationSpeed_eachCycle.append(copy.deepcopy(temp_propagationSpeed))

        ####################  plot results3.2  ####################
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        for i in range(2,len(x_cm_eachCycle_speed)):
            ax.scatter(x_cm_eachCycle_speed[i], propagationSpeed_eachCycle[i], \
        	           s=30, color='darkslateblue', alpha=0.7, linewidths="1",  edgecolor='darkslateblue')

        ####################  graph style  ####################
        ax.set_xlabel('x (cm)', FontWeight='bold', FontSize=24)
        ax.set_ylabel('Progation speed (cm/s)', FontWeight='bold', FontSize=24)
        ax.set_ylim(-220, 20)
        ax.tick_params(labelsize=24)
        #ax.set_ylim(,)
        #ax.set_xlim(,)
        # ax.set_title('FREI', FontWeight='bold')
        plt.subplots_adjust(left=0.18, right=0.86, bottom=0.14, top=0.94)
        plt.savefig('./{}/result3propagationSpeed.png'.format(self.saveDir))
        plt.show()

    def expPropagatingSpeed(self):
        maxLoc, maxValue = self.detectMaximumPeak(self.dfList)
        maxAll = max(maxValue)
        quenchingPara = 0.5
        for i in range(len(maxValue)):
            if maxValue[i] < quenchingPara*maxAll:
                maxValue[i] = 0
        #print(maxValue)

        ######### detect ignition and extinction point ##########
        ignitionNum = []
        ignitionTime = []
        ignitionPosition = []
        extinctionNum = []
        extinctionTime = []
        extinctionPosition = []
        for i in range(len(maxValue)-1):
            if maxValue[i] == 0 and maxValue[i+1] != 0:
                ignitionNum.append(i+1)
                ignitionTime.append((i+1)*self.deltaT)
                ignitionPosition.append(maxLoc[i+1])
            if maxValue[i] != 0 and maxValue[i+1] == 0:
                extinctionNum.append(i+1)
                extinctionTime.append((i+1)*self.deltaT)
                extinctionPosition.append(maxLoc[i])
        #print(ignitionPosition)
        #for num in ignitionNum:
        #	print(maxLoc[num])

        ############### obtain cycle of FREI ##################
        numIntervalIgnition= []
        numIntervalExtinction= []
        timeIntervalIgnition= []
        timeIntervalExtinction= []
        for i in range(len(ignitionNum)-1):
            numIntervalIgnition.append(ignitionNum[i+1]-ignitionNum[i])
            timeIntervalIgnition.append(ignitionTime[i+1]-ignitionTime[i])
        for i in range(len(extinctionNum)-1):
            numIntervalExtinction.append(extinctionNum[i+1]-extinctionNum[i])
            timeIntervalExtinction.append(extinctionTime[i+1]-extinctionTime[i])

        ############  obtain time and distance from ignition to extinction  ############
        flamePropagetingTime=[]
        flamePropagetingDistance=[]
        flameQuenchingTime=[]
        flameQuenchingDistance=[]
        if ignitionNum[0] > extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(len(ignitionNum)-1):
                    flamePropagetingTime.append(extinctionTime[i+1] - ignitionTime[i])
                    flamePropagetingDistance.append(ignitionPosition[i] - extinctionPosition[i+1])
                for i in range(len(ignitionNum)):
                    flameQuenchingTime.append(ignitionTime[i] - extinctionTime[i])
                    flameQuenchingDistance.append(ignitionPosition[i] - extinctionPosition[i])
            else:
                for i in range(len(ignitionNum)):
                    flamePropagetingTime.append(extinctionTime[i+1] - ignitionTime[i])
                    flamePropagetingDistance.append(ignitionPosition[i] - extinctionPosition[i+1])
                    flameQuenchingTime.append(ignitionTime[i] - extinctionTime[i])
                    flameQuenchingDistance.append(ignitionPosition[i] - extinctionPosition[i])
        elif ignitionNum[0] < extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(len(extinctionNum)):
                    flamePropagetingTime.append(extinctionTime[i] - ignitionTime[i])
                    flamePropagetingDistance.append(ignitionPosition[i] - extinctionPosition[i])
                for i in range(len(extinctionNum)-1):
                    flameQuenchingTime.append(ignitionTime[i+1] - extinctionTime[i])
                    flameQuenchingDistance.append(ignitionPosition[i+1] - extinctionPosition[i])
            else:
                for i in range(len(extinctionNum)):
                    flamePropagetingTime.append(extinctionTime[i] - ignitionTime[i])
                    flamePropagetingDistance.append(ignitionPosition[i] - extinctionPosition[i])
                    flameQuenchingTime.append(ignitionTime[i+1] - extinctionTime[i])
                    flameQuenchingDistance.append(ignitionPosition[i+1] - extinctionPosition[i])

        ############  obtain average propagation speed  ############
        averagePropagationSpeed = []
        for i in range(len(flamePropagetingTime)):
            averagePropagationSpeed.append(-flamePropagetingDistance[i]/flamePropagetingTime[i])

        #########  obtain average convection speed when quenching  #########
        averageConvectionSpeed = []
        for i in range(len(flameQuenchingTime)):
            averageConvectionSpeed.append(flameQuenchingDistance[i]/flameQuenchingTime[i])

        ###################  save results2  ####################
        f2 = open('./{}/result2.csv'.format(self.saveDir),'w')
        f2.write("num,igTime,igLoc,igTw,Tig,exTime,exLoc,exTw,Tex,ig_exTime,ig_exDist,Vave,ex_igTime,ex_igDist,Vconv\n")
        if len(timeIntervalIgnition) < len(timeIntervalExtinction):
            loopNum = len(timeIntervalIgnition)
        else:
            loopNum = len(timeIntervalExtinction)
        for i in range(loopNum):
            out1=ignitionTime[i]
            out2=ignitionPosition[i]
            out3=self.TwallFunc(out2)
            out4=timeIntervalIgnition[i]
            out5=extinctionTime[i]
            out6=extinctionPosition[i]
            out7=self.TwallFunc(out6)
            out8=timeIntervalExtinction[i]
            out9=flamePropagetingTime[i]
            out10=flamePropagetingDistance[i]
            out11=averagePropagationSpeed[i]
            out12=flameQuenchingTime[i]
            out13=flameQuenchingDistance[i]
            out14=averageConvectionSpeed[i]
            f2.write("{0:},{1:.5f},{2:.3f},{3:.2f},{4:.6f},{5:.5f},{6:.3f},{7:.2f},{8:.6f},{9:.5f},{10:.4f},{11:.3f},{12:.5f},{13:.3f},{14:.3f}\n"\
                    .format(i,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11,out12,out13,out14))
            f2.flush()
        f2.close()

        ###########  delete list while quenching  ############
        t=[]
        [t.append(i*self.deltaT) for i in range(len(maxLoc))]
        delNum = []
        delNum.extend(list(range(ignitionNum[0])))
        if ignitionNum[0] > extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(1,len(ignitionNum)):
                    delNum.extend(list(range(extinctionNum[i],ignitionNum[i])))
            else:
                for i in range(1,len(ignitionNum)):
                    delNum.extend(list(range(extinctionNum[i],ignitionNum[i])))
                delNum.extend(list(range(extinctionNum[len(ignitionNum)-1],len(maxValue))))
        elif ignitionNum[0] < extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(len(extinctionNum)-1):
                    delNum.extend(list(range(extinctionNum[i],ignitionNum[i+1])))
                delNum.extend(list(range(extinctionNum[len(extinctionNum)-1],len(maxValue))))
            else:
                for i in range(len(extinctionNum)):
                    delNum.extend(list(range(extinctionNum[i],ignitionNum[i+1])))
        dellist = lambda items, indexes: [item for index, item in enumerate(items) if index not in indexes]
        t_propagating = dellist(t, delNum)
        maxLoc_propagating = dellist(maxLoc, delNum)
        maxValue_propagating = dellist(maxValue, delNum)

        ############  obtain propagation speed  ############
        propagationSpeed = []
        threshold_ignition_extinction = 1.0
        for i in range(len(maxLoc_propagating)-1):
            propagationSpeed.append((maxLoc_propagating[i+1]-maxLoc_propagating[i])/self.deltaT)

        ###################  save results3  ####################
        f3 = open('./{}/result3.csv'.format(self.saveDir),'w')
        f3.write("num,t,x,Tw,hrr,v\n")
        for i in range(len(propagationSpeed)-1):
            t_out=t_propagating[i]
            x_out=maxLoc_propagating[i]
            Twall_out=self.TwallFunc(x_out)
            max_out=maxValue_propagating[i]
            speed_out=propagationSpeed[i]
            f3.write("{0:},{1:.4f},{2:.3f},{3:.2f},{4:.1f},{4:}\n".format(i, t_out, x_out, Twall_out, max_out, speed_out))
            f3.flush()
        f3.close()

        ###########  obtain list while propagating  ############
        xEachCycle=[]
        maxValueEachCycle=[]
        if ignitionNum[0] > extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(0,len(ignitionNum)-1):
                    xEachCycle.append(maxLoc[ignitionNum[i]:extinctionNum[i+1]])
                    maxValueEachCycle.append(list(maxValue[ignitionNum[i]:extinctionNum[i+1]]))
            else:
                for i in range(0,len(ignitionNum)):
                    xEachCycle.append(maxLoc[ignitionNum[i]:extinctionNum[i+1]])
                    maxValueEachCycle.append(list(maxValue[ignitionNum[i]:extinctionNum[i+1]]))
        elif ignitionNum[0] < extinctionNum[0]:
            if len(ignitionNum) == len(extinctionNum):
                for i in range(len(extinctionNum)):
                    xEachCycle.append(maxLoc[ignitionNum[i]:extinctionNum[i]])
                    maxValueEachCycle.append(list(maxValue[ignitionNum[i]:extinctionNum[i]]))
            else:
                for i in range(len(extinctionNum)):
                # for i in range(len(extinctionNum)+1):  #from ignition to ... before extinction
                    xEachCycle.append(maxLoc[ignitionNum[i]:extinctionNum[i]])
                    maxValueEachCycle.append(list(maxValue[ignitionNum[i]:extinctionNum[i]]))

        ############  obtain propagation speed each cycle  ############
        x_cm_eachCycle_speed = []
        max_eachCycle_speed = []
        propagationSpeed_eachCycle = []
        temp_x = []
        temp_maxValue = []
        temp_propagationSpeed = []
        for i in range(len(xEachCycle)):
            temp_x.clear()
            temp_maxValue.clear()
            temp_propagationSpeed.clear()
            for j in range(len(xEachCycle[i])-1):
                temp_x.append(xEachCycle[i][j])
                temp_maxValue.append(maxValueEachCycle[i][j])
                temp_propagationSpeed.append((xEachCycle[i][j+1]-xEachCycle[i][j])/self.deltaT)
            x_cm_eachCycle_speed.append(copy.deepcopy(temp_x))
            max_eachCycle_speed.append(copy.deepcopy(temp_maxValue))
            propagationSpeed_eachCycle.append(copy.deepcopy(temp_propagationSpeed))

        ####################  plot results3.2  ####################
        fig32, ax321 = plt.subplots(nrows=1, ncols=1, figsize=(11, 8))
        #ax322 = ax321.twinx()
        #cmap1 = plt.get_cmap("jet")
        #cmap2 = plt.get_cmap("jet")
        #cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(2,len(x_cm_eachCycle_speed)):
            # ax321.plot(x_cm_eachCycle_speed[i], propagationSpeed_eachCycle[i], \
        #	           color=cmap1(i/len(x_cm_eachCycle_speed)), alpha=1.0, linestyle=':')
        #	           color='purple', alpha=1.0, linestyle=':')
            # ax322.plot(x_cm_eachCycle_speed[i], max_eachCycle_speed[i], \
        #	           color=cmap2(i/len(x_cm_eachCycle_speed)), alpha=1.0, linestyle='-', label=i)
        #	           color='purple', alpha=1.0, linestyle='-', label=i)
            for j in range(len(x_cm_eachCycle_speed[i])):
                ax321.scatter(x_cm_eachCycle_speed[i][j], propagationSpeed_eachCycle[i][j], \
        	              s=30, color='darkslateblue', alpha=0.7, \
        #                     s=30, color=cmap1(i/len(x_cm_eachCycle_speed)), alpha=0.7, \
                              linewidths="1",  edgecolor='darkslateblue')
        #		              linewidths="1",  edgecolor=cmap2(i/len(x_cm_eachCycle_speed)))
        #		ax322.scatter(x_cm_eachCycle_speed[i][j], max_eachCycle_speed[i][j], \
        #		              s=30, color='firebrick', alpha=1.0, \
        #		              s=30, color=cmap2(i/len(x_cm_eachCycle_speed)), alpha=1.0, \
        #		              linewidths="1", edgecolor='firebrick')
        #		              linewidths="1", edgecolor=cmap2(i/len(x_cm_eachCycle_speed)))
        ####################  graph style  ####################
        ax321.set_xlabel('x (cm)', FontWeight='bold', FontSize=24)
        ax321.set_ylabel('Progation speed (cm/s)', FontWeight='bold', FontSize=24)
        # ax321.set_ylim(-800,300)
        ax321.set_ylim(-220, 20)
        ax321.tick_params(labelsize=24)
        #ax322.set_ylabel('Heat release rate (J/m^3-s)', FontWeight='bold', FontSize=24)
        #ax322.legend(loc='upper right')
        #ax322.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        #ax322.ticklabel_format(style="sci",  axis="y",scilimits=(0,0))
        #ax322.yaxis.offsetText.set_fontsize(24)
        #ax322.tick_params(labelsize=24)
        #ax321.set_ylim(,)
        #ax321.set_xlim(,)
        #ax321.set_title('FREI', FontWeight='bold')
        plt.subplots_adjust(left=0.18, right=0.86, bottom=0.14, top=0.94)
        plt.savefig('./{}/result3propagationSpeed.png'.format(self.saveDir))
        plt.show()

        
        
