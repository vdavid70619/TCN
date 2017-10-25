import json
import numpy as np
import cPickle as pickle

def iou(test, target):

    tt1 = np.maximum(target[0], test[0])
    tt2 = np.minimum(target[1], test[1])

    # Non-negative overlap score
    intersection = (tt2 - tt1).clip(0)
    union = ((test[1] - test[0]) +
             (target[1] - target[0]) -
             intersection)
    # Compute overlap as the ratio of the intersection
    # over union of two segments at the frame level.
    iou = intersection / union
    return iou

# support customized levels, the level means
def gen_temporal_pyramid(data, levels=[1./32, 1./16, 1./8, 1./4, 1./2, 1.], thr_ious=[.4, .45, .5, .55, .6, .7], overlap=0.5, SUBSET='validation'):
    video_id_fmt = 'v_{}'
    proposals = {}
    for vid, vitem in data.iteritems():
        if vitem['subset'] != SUBSET:
            continue

        gt_segments = [x['segment'] for x in vitem['annotations']]
        gt_labels = [x['class'] for x in vitem['annotations']]
        duration = vitem['duration']
        split = duration*levels[0]
        anchor = np.arange(split/2, duration-split/2, split*overlap)
        proposals[vid] = []
        for p in anchor:

            one_pyramid = []
            gt_index = -1
            for i in range(len(levels)):
                split = duration*levels[i]
                # handle multiple gts case, choose the one close to the level proposoal
                ious = [iou([max(0,p-split/2), min(duration,p+split/2)], gt) for gt in gt_segments]
                gt_index = np.argmax(ious)

                # also include the nearest class label for further usage
                one_pyramid.append([max(0,p-split/2), min(duration,p+split/2),
                                    ious[gt_index],
                                    gt_labels[gt_index] if ious[gt_index]>=thr_ious[i] else 0
                                    ])
                # one_pyramid.append([p-split/2, p+split/2,
                #                     ious[gt_index],
                #                     gt_labels[gt_index] if ious[gt_index]>=thr_ious[i] else 0
                #                     ])
            proposals[vid].append(one_pyramid)

    return proposals

def test_temporal_pyramid(data, levels=[1./32, 1./16, 1./8, 1./4, 1./2, 1.], overlap=0.5, SUBSET='testing'):
    video_id_fmt = 'v_{}'
    proposals = {}
    for vid, vitem in data.iteritems():
        if vitem['subset'] != SUBSET:
            continue

        duration = vitem['duration']
        split = duration*levels[0]
        anchor = np.arange(split/2, duration-split/2, split*overlap)
        proposals[vid] = []
        for p in anchor:

            one_pyramid = []
            gt_index = -1
            for i in range(len(levels)):
                split = duration*levels[i]
                # also include the nearest class label for further usage
                one_pyramid.append([max(0,p-split/2), min(duration,p+split/2),0,0])
                # one_pyramid.append([p-split/2, p+split/2,0,0])
            proposals[vid].append(one_pyramid)

    return proposals

if __name__ == "__main__":
    ANNOTATION_FILE = 'actNet200-V1-3.pkl'
    with open(ANNOTATION_FILE,'rb') as f:
         data = pickle.load(f)['database']

    train_proposals = gen_temporal_pyramid(data, SUBSET='training')
    val_proposals = gen_temporal_pyramid(data, SUBSET='validation')
    test_proposals = test_temporal_pyramid(data)

    pickle.dump(train_proposals, open("train_proposals.pkl", "wb"))
    pickle.dump(val_proposals, open("val_proposals.pkl", "wb"))
    pickle.dump(test_proposals, open("test_proposals.pkl", "wb"))
    debug = 0