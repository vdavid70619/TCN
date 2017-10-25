import numpy as np

def extract(feat, num_frame, duration, start, end, K, zero_padding=False, repeat=True):
    fps = num_frame / duration
    numfeat = feat.shape[0]
    indexes = np.asarray(np.arange(numfeat))

    startframe = round(start*fps)/round(fps*0.4)
    endframe = round(end*fps)/round(fps*0.4)
    #get in range indexes
    inrange = np.nonzero([np.logical_and(indexes<=endframe, indexes>=startframe)])[1]

    if len(inrange)==0:
        return None

    start_out = max(0-start,0)
    end_out = max(end-duration,0)
    if (start_out+end_out)>0 and zero_padding:
        start_zeros = int(round(start_out*K/(end-start)))
        end_zeros = int(round(end_out*K/(end-start)))

        nselect = K-start_zeros-end_zeros
        if len(inrange)<nselect and not repeat:
            selected = inrange
        else:
            inds = np.floor(np.arange(0,len(inrange)-1e-6,float(len(inrange))/nselect)+float(len(inrange))/(2*nselect)).astype(int)
            selected = inrange[inds]

        data = np.vstack((np.zeros((start_zeros, feat.shape[1])), feat[selected,:], np.zeros((end_zeros, feat.shape[1]))))
    else:
        if len(inrange)<K and not repeat:
            selected = inrange
        else:
            inds = np.floor(np.arange(0,len(inrange)-1e-6,float(len(inrange))/K)+float(len(inrange))/(2*K)).astype(int)
            selected = inrange[inds]

        data = feat[selected,:]

    return data
