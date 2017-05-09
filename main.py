import tkinter as tk
from tkinter import filedialog
import dicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import pylab
import os

# =================================================================================
#                                Get CT Image
# =================================================================================
def get_pixels_HU(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope     = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

root = tk.Tk()
root.withdraw()
dcmPath = filedialog.askopenfilename()
print(dcmPath)
slice=dicom.read_file(dcmPath)

pixel_raw = slice.pixel_array
pixel_HU = get_pixels_HU([slice])[0]
ct_data = pixel_HU

featurePath=os.path.basename(dcmPath)+".csv"
plt.switch_backend('Qt4Agg')
plt.figure(figsize=(24, 17),num="Python Image Features: Allright Reserved by XGlab - Current File=%s"%featurePath)

thismanager = plt.get_current_fig_manager()
from PyQt4 import QtGui
thismanager.window.setWindowIcon(QtGui.QIcon((os.path.join('res','shepherd.png'))))

# =================================================================================
#                                Extracted Window
# =================================================================================
from RadiomicFeatures.Feature235 import featuresV2
class ExtractedWindow:
    def __init__(self, topleft, downright, slice):
        self.topleft, self.downright=topleft, downright
        x0,y0 = int(topleft[0]),int(topleft[1])
        x1,y1 = int(downright[0]),int(downright[1])
        self.pixel=slice[y0:y1, x0:x1].astype(np.float64)

    def getFeatures(self):
        return featuresV2(self.pixel)

    def __str__(self):
        return "(%.2f;%.2f)->(%.2f;%.2f)"% (self.topleft[0],self.topleft[1],self.downright[0],self.downright[1])

extractedWindowHub = []


# =================================================================================
#                                   Text Box
# =================================================================================
txtAx = plt.subplot2grid( (3,4), (0,3), rowspan=2,colspan=1)
txtAx.set_title("windows extracted", fontsize=30)
txtAx.title.set_position([.5, 1.05])
xlim,ylim=(10,30)
txtAx.axis([0, xlim, 0, ylim])
textIndent=0.75
linePos=29

def addMessage(str):
    global linePos
    global txtAx
    txtAx.text(textIndent, linePos, str, fontsize=12)
    linePos= (linePos-1 +ylim) % ylim

# =================================================================================
#                                Rectangle Selector
# =================================================================================
from matplotlib.widgets import RectangleSelector
import numpy as np

def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    toggle_selector.RS.set_active(False) # disable 'Q' to avoid unintended rectangle saving
    print("   未选择")



def toggle_selector(event):
    global extractedWindowHub

    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        #print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
        xmin,ymin=toggle_selector.RS.extents[0], toggle_selector.RS.extents[2]
        xmax,ymax=toggle_selector.RS.extents[1], toggle_selector.RS.extents[3]
        #xmin,ymin=(374.36,282.10)
        #xmax,ymax=(384.33,298.72)
        if int(xmin)==int(xmax) or int(ymin)==int(ymax):
            print("   未选择")
            return
        print("   选择成功: 左上(%.2f,%.2f) 右下(%.2f,%.2f)"%(xmin,ymin,xmax,ymax))
        #selectedWin=ExtractedWindow(topleft=(xmin,ymax), downright=(xmax,ymin), slice=ct_data)
        selectedWin=ExtractedWindow(topleft=(xmin,ymin), downright=(xmax,ymax), slice=ct_data)
        extractedWindowHub.append(selectedWin)
        extractedWindowHub = list(set(extractedWindowHub))
        addMessage(str(selectedWin))

    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        #print(' RectangleSelector activated.')
        print('>  进入选择模式')
        toggle_selector.RS.set_active(True)


imgAx = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=3)
imgAx.set_xlabel('X-Axis')
imgAx.set_ylabel('Y-Axis')
imgAx.set_title('Press A to select, press Q when finished',fontsize=30)
imgAx.title.set_position([.5, 1.05])
line, = imgAx.plot([], [], linestyle="none", marker="o", color="r")
plt.imshow(ct_data, cmap=pylab.cm.bone,interpolation='nearest')

toggle_selector.RS = RectangleSelector(imgAx, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=False)
toggle_selector.RS.set_active(False)
plt.connect('key_press_event', toggle_selector)


# =================================================================================
#                        Calculate Features and Output
# =================================================================================
from matplotlib.widgets import Button
from RadiomicFeatures.Feature235 import GetFGNameList

def onCalcClicked(event):
    global extractedWindowHub
    if extractedWindowHub==[]:
        print(">  请先选择待分析区域")
        return
    print("=================================")
    csvLines=[]
    for win in extractedWindowHub:
        print(win)
        featureVector=win.getFeatures()
        lineTxt=[str(win)] + featureVector
        csvLines.append(lineTxt)

    try:
        os.remove(featurePath)
    except OSError:
        pass
    csvHeader="WindowPos,"+ ",".join(GetFGNameList())
    np.savetxt(featurePath, csvLines,fmt='%s',delimiter=',',header=csvHeader)
    print("   特征提取完成")
    global txtAx
    txtAx.figure.canvas.draw()

def onResetClicked(event):
    global extractedWindowHub
    extractedWindowHub=[]
    print(">  归零")
    btnReset.ax.figure.canvas.draw()
    global txtAx
    global linePos
    txtAx.clear()
    txtAx.set_title("Windows Extracted", fontsize=30)
    txtAx.title.set_position([.5, 1.05])
    txtAx.figure.canvas.draw()
    linePos=29

axCalc = plt.axes([0.75, 0.23, 0.09, 0.04])
axReset = plt.axes([0.75, 0.17, 0.09, 0.04])

# CAUTION! We must specify color and hovercolor with the same value explicitly,
# otherwise the hover-over event will cause a unwanted redraw.
btnCalc = Button(axCalc,   'Compute',color='0.85', hovercolor='0.85')
btnCalc.label.set_fontsize(25)
# And we hope to clear the previous rectangles here, so we choose a sligly different hovercolor
btnReset = Button(axReset, 'Reset',color='0.85', hovercolor='0.85')
btnReset.label.set_fontsize(25)
btnCalc.on_clicked(onCalcClicked)
btnReset.on_clicked(onResetClicked)

plt.show()
