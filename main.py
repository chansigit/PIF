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
from RadiomicFeatures.Feature440 import GetFGName
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


# =================================================================================
#                                Extracted Window
# =================================================================================
from RadiomicFeatures.Feature440 import featuresV2
class ExtractedWindow:
    def __init__(self, topleft, downright, slice):
        self.topleft, self.downright=topleft, downright
        x0,y0 = int(topleft[0]),int(topleft[1])
        x1,y1 = int(downright[0]),int(downright[1])
        self.pixel=slice[y0:y1, x0:x1].astype(np.float32)

    def getFeatures(self):
        return featuresV2(self.pixel)

    def __str__(self):
        return "(%.2f;%.2f)->(%.2f;%.2f)"% (self.topleft[0],self.topleft[1],self.downright[0],self.downright[1])


# =================================================================================
#                                Rectangle Selector
# =================================================================================
extractedWindowHub = []
from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt

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
        print("   选择成功: 左上(%.2f,%.2f) 右下(%.2f,%.2f)"%(xmin,ymin,xmax,ymax))
        #selectedWin=ExtractedWindow(topleft=(xmin,ymax), downright=(xmax,ymin), slice=ct_data)
        selectedWin=ExtractedWindow(topleft=(xmin,ymin), downright=(xmax,ymax), slice=ct_data)
        extractedWindowHub.append(selectedWin)
        extractedWindowHub = list(set(extractedWindowHub))

    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        #print(' RectangleSelector activated.')
        print('>  进入选择模式')
        toggle_selector.RS.set_active(True)



fig = plt.figure(figsize=(12,8), num="Python Image Features: Allright Reserved by XGlab - Current File=%s"%featurePath)
thismanager = plt.get_current_fig_manager()
thismanager.window.wm_iconbitmap("microscope.ico")
ax = fig.add_subplot(111)
ax.set_title('Press A to select, press Q when finished')
line, = ax.plot([], [], linestyle="none", marker="o", color="r")
plt.imshow(ct_data, cmap=pylab.cm.bone,interpolation='nearest')

toggle_selector.RS = RectangleSelector(ax, line_select_callback,
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
        #print(len(featureVector))
        #print(featureVector)
        lineTxt=[str(win)] + featureVector
        csvLines.append(lineTxt)

    try:
        os.remove(featurePath)
    except OSError:
        pass

    csvHeader="WindowPos,"+GetFGName()

    np.savetxt(featurePath, csvLines,fmt='%s',delimiter=',',header=csvHeader)
    print("   特征提取完成")


def onResetClicked(event):
    global extractedWindowHub
    extractedWindowHub=[]
    print(">  归零")
    btnReset.ax.figure.canvas.draw()


axCalc = plt.axes([0.35, 0.03, 0.07, 0.035])
axReset = plt.axes([0.55, 0.03, 0.07, 0.035])

# CAUTION! We must specify color and hovercolor with the same value explicitly,
# otherwise the hover-over event will cause a unwanted redraw.
btnCalc = Button(axCalc,   'Compute',color='0.85', hovercolor='0.85')
# And we hope to clear the previous rectangles here, so we choose a sligly different hovercolor
btnReset = Button(axReset, 'Reset',color='0.85', hovercolor='0.85')
btnCalc.on_clicked(onCalcClicked)
btnReset.on_clicked(onResetClicked)

plt.show()

pylab.show()