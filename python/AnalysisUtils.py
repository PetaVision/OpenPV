import PVPython as pv
import matplotlib.pyplot as plt
import math
import numpy as np

def printProbeTable(pvContext, energyProbe, errorProbe, activityProbe, adaptProbe):
    # Move cursor to top left
    print('\033[1;1H', end='')
    en_list = pv.getProbeValues(pvContext, energyProbe)
    l2_list = pv.getProbeValues(pvContext, errorProbe)
    l1_list = pv.getProbeValues(pvContext, activityProbe)
    ad_list = pv.getProbeValues(pvContext, adaptProbe)
    print("[Batch]\t[Energy]\t[Error]\t[Activity]\t[Timescale]")
    for i in range(len(en_list)):
        print("%d:\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f" %
                (i,
                 round(en_list[i], 2),
                 round(l2_list[i], 2),
                 round(l1_list[i], 2),
                 round(ad_list[i], 2)))
    print("Avg:\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f" %
            (round(sum(en_list) / len(en_list), 2),
             round(sum(l2_list) / len(l2_list), 2),
             round(sum(l1_list) / len(l1_list), 2),
             round(sum(ad_list) / len(ad_list), 2)))

def plotRecons(pvContext, imageName, errorName, reconName, fig=1):
    img = pv.getLayerActivity(pvContext, imageName)
    err = pv.getLayerActivity(pvContext, errorName)
    rec = pv.getLayerActivity(pvContext, reconName)
    im_row = np.concatenate((img, err, rec), axis=1)
    full = np.concatenate(np.split(im_row, np.shape(im_row)[0], 0), axis=2)
    # normalize to the range of the input image
    if np.min(full) != np.max(full):
        full = (full - np.min(img)) / (np.max(img) - np.min(img))
    plt.figure(fig)
    plt.clf()
    plt.imshow(np.clip(full[0], 0.0, 1.0))
    plt.draw()
    plt.pause(0.001)

def plotWeights(pvContext, connName, fig=2):
    weights = pv.getConnectionWeights(pvContext, connName)
    weights = np.split(weights, np.shape(weights)[0], 0)
    sq = math.sqrt(len(weights))
    w = math.ceil(np.shape(weights[0])[2] * sq)
    h = math.ceil(np.shape(weights[0])[1] * sq)
    ratio = h / w
    per_row = math.ceil(sq * ratio)
    rows = []
    for i in range(math.ceil(len(weights)/per_row)):
        rows.append(np.concatenate(weights[i*per_row:min(len(weights), (i+1)*per_row)], axis=2))
    w_full = np.concatenate(rows, axis=1)
    if np.min(w_full) != np.max(w_full):
        w_full = (w_full - np.min(w_full)) / (np.max(w_full) - np.min(w_full))
    plt.figure(fig)
    plt.clf()
    plt.imshow(w_full[0])
    plt.draw()
    plt.pause(0.001)


