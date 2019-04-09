# trace generated using paraview version 5.5.2
#### import the simple module from the paraview
from paraview.simple import *
# import module
import os
import sys
from sys import argv


class PvbatchMFRPostProcess():
    def __init__(self, readPath):
        self.readPath = readPath
        print(self.readPath)
        
        #### disable automatic camera reset on 'Show'
        paraview.simple._DisableFirstRenderCameraReset()
        # get active view
        self.renderView1 = GetActiveViewOrCreate('RenderView')
        # uncomment following to set a specific view size
        # renderView1.ViewSize = [3430, 2192]
        # Properties modified on renderView1
        self.renderView1.Background = [1.0, 1.0, 1.0]
        # create a new 'OpenFOAMReader'
        self.xfoam = OpenFOAMReader(FileName=self.readPath)
        self.xfoam.MeshRegions = ['internalMesh']
        # get display properties
        self.xfoamDisplay = GetDisplayProperties(self.xfoam, view=self.renderView1)
        # Properties modified on xfoamDisplay
        self.xfoamDisplay.EdgeColor = [1.0, 1.0, 1.0]
        # Properties modified on renderView1
        self.renderView1.OrientationAxesVisibility = 0
        # hide color bar/color legend
        self.xfoamDisplay.SetScalarBarVisibility(self.renderView1, False)

    def __my_makedirs(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def pvbatch1dCsv(self, p1, p2):
        print("mfrPostProcess pvbatch1dCsv")
        saveName='profile'
        line = [p1,p2]
        print(self.xfoam.CellArrays)
        # create a new 'Plot Over Line'
        plotOverLine1 = PlotOverLine(Input=self.xfoam, Source='High Resolution Line Source')
        # init the 'High Resolution Line Source' selected for 'Source'
        plotOverLine1.Source.Point1 = line[0]
        plotOverLine1.Source.Point2 = line[1]
        # show data in view
        plotOverLine1Display = Show(plotOverLine1, self.renderView1)
        # Create a new 'Line Chart View'
        lineChartView1 = CreateView('XYChartView')
        lineChartView1.ViewSize = [890, 2188]
        # get layout
        layout1 = GetLayout()
        # place view in the layout
        layout1.AssignView(2, lineChartView1)
        # show data in view
        plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1)
        # trace defaults for the display properties.
        plotOverLine1Display_1.CompositeDataSetIndex = [0]
        plotOverLine1Display_1.UseIndexForXAxis = 0
        plotOverLine1Display_1.XArrayName = 'arc_length'
        # update the view to ensure updated data information
        self.renderView1.Update()
        # update the view to ensure updated data information
        lineChartView1.Update()
        # save data
        print('line0y={0}, line1y={1}'.format(line[0][1], line[1][1]))
        #os.makedirs('csv1d')
        saveDir = 'csv1d_y'+str(line[0][1])
        self.__my_makedirs(saveDir)
        SaveData('{0}/{1}.csv'.format(saveDir, saveName), proxy=plotOverLine1, UseScientificNotation=1, WriteTimeSteps=1)
        #### uncomment the following to render all views
        # RenderAllViews()
        # alternatively, if you want to write images, you can use SaveScreenshot(...).



