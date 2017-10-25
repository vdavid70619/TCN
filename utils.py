import cPickle as pickle
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
import json
import urllib2

API = 'http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/challenge16/api.py'

def get_blocked_videos(api=API):
    api_url = '{}?action=get_blocked'.format(api)
    req = urllib2.Request(api_url)
    response = urllib2.urlopen(req)
    return json.loads(response.read())

def softmax(raw_score, T=1):
    exp_s = np.exp((raw_score - raw_score.max(axis=-1)[..., None])*T)
    sum_s = exp_s.sum(axis=-1)
    return exp_s / sum_s[..., None]

def sliding_window_aggregation_func(score, spans=[1, 2, 4, 8, 16], overlap=0.2, norm=True, fps=1):
    frm_max = score.max(axis=1)
    slide_score = []

    def top_k_pool(scores, k):
        return np.sort(scores, axis=0)[-k:, :].mean(axis=0)

    for t_span in spans:
        span = t_span * fps
        step = int(np.ceil(span * (1-overlap)))
        local_agg = [frm_max[i: i+span].max(axis=0) for i in xrange(0, frm_max.shape[0], step)]
        k = max(15, len(local_agg)/4)
        slide_score.append(top_k_pool(np.array(local_agg), k))

    out_score = np.mean(slide_score, axis=0)

    if norm:
        return softmax(out_score)
    else:
        return out_score

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def segment_iou(target_segment, candidate_segments):
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def batch_segment_iou(target_segments, test_segments):
    if target_segments.ndim != 2 or test_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    m, n = target_segments.shape[0], test_segments.shape[0]
    iou = np.empty((m, n))
    for i in xrange(m):
        tt1 = np.maximum(target_segments[i, 0], test_segments[:, 0])
        tt2 = np.minimum(target_segments[i, 1], test_segments[:, 1])

        # Non-negative overlap score
        intersection = (tt2 - tt1).clip(0)
        union = ((test_segments[:, 1] - test_segments[:, 0]) +
                 (target_segments[i, 1] - target_segments[i, 0]) -
                 intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments at the frame level.
        iou[i, :] = intersection / union
    return iou

def nms(proposals, scores, thresh):
    x1 = proposals[:, 0]
    x2 = proposals[:, 1]

    areas = (x2 - x1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[1:][inds]

    return keep

def nms_all(proposals, topK=20, nms_thor=0.45, remove_background=False):
    nms_proposals = {}
    for vid, proposal in proposals.iteritems():
        proposal = np.array(proposal)
        if proposal.ndim>2:
            proposal = proposal.reshape(proposal.shape[0]*proposal.shape[1], proposal.shape[2])
        if remove_background:
            proposal = proposal[proposal[:,3]>0,:]

        keep = nms(proposal, proposal[:, 2], nms_thor)
        nms_proposals[vid] = proposal[keep[:topK], :]
    return nms_proposals


def gen_json(proposals, id2name):
    output = {'results': {}, 'version': "VERSION 1.3", 'external_data': {'used': False, 'details': ""}}
    activity_names = id2name

    for vid, proposal_per_vid in proposals.iteritems():
        proposal = np.asarray(proposal_per_vid)
        if len(proposal) > 0:
            if proposal.ndim>2:
                proposal = proposal.reshape((proposal.shape[0] * proposal.shape[1], proposal.shape[2]))

            output['results'][vid] = []
            for i in range(proposal.shape[0]):
                output['results'][vid].append({'segment': [float(proposal[i, 0]), float(proposal[i, 1])],
                                               'score': float(proposal[i, 2]),
                                               'label': activity_names[proposal[i, 3]]})
    return output

def recall_vs_iou_thresholds(proposal_df, df, iou_threshold=np.arange(0.1, 1.1, 0.1)):
    vds = proposal_df['video-id'].unique()
    score_lst = []
    # Compute iou score
    for i, v in enumerate(vds):
        # Proposals
        idx = proposal_df['video-id'] == v
        this_df = proposal_df.loc[idx]
        proposals = np.stack((this_df['s-init'],
                              this_df['s-end']), axis=-1)

        # Sort proposals
        idx = this_df['score'].argsort()[::-1]
        proposals = proposals[idx, :]

        # Annotations
        jdx = df['video-id'] == v
        ann_df = df.loc[jdx]
        annotations = np.stack((ann_df['s-init'],
                                ann_df['s-end']), axis=-1)
        if proposals.ndim == 1:
            proposals = proposals[np.newaxis, :]
        score_lst.append(batch_segment_iou(annotations, proposals))
        if not (i+1)%500:
            print 'Scored videos: {}'.format(i+1)
    matches = np.zeros((vds.shape[0], iou_threshold.shape[0]))
    pos = np.zeros(vds.shape[0])
    # Matching
    recall = np.empty(iou_threshold.shape[0])
    for cidx, this_iou in enumerate(iou_threshold):
        # Score analysis per video.
        for i, sc in enumerate(score_lst):
            pos[i] = sc.shape[0] # Positives per video.
            lmt = int(sc.shape[1])
            matches[i, cidx] = ((sc[:, :lmt] >= this_iou).sum(axis=1) > 0).sum()
        this_recall = matches[:, cidx].sum() / pos.sum()
        recall[cidx] = this_recall
    return recall

def convert(proposal):
    pdp = pd.DataFrame({'s-init': proposal['s-init'],
                          's-end': proposal['s-end'],
                          'score': proposal['score'],
                          'video-id': proposal['video-id']})
    return pdp

def get_gt(data=None, gt_file='actNet200-V1-3.pkl', SUBSET='validation'):
    if data is None:
        with open(gt_file, 'r') as fobj:
            data = pickle.load(fobj)['database']

    video_id_fmt = 'v_{}'
    video_dur = []
    gt_s_init, gt_s_end, video_id = [], [], []
    for vid, vitem in data.iteritems():
        video_dur.append(vitem['duration'])
        if vitem['subset'] != SUBSET:
            continue
        for ann in vitem['annotations']:
            gt_s_init.append(ann['segment'][0])
            gt_s_end.append(ann['segment'][1])
            video_id.append(video_id_fmt.format(vid))
    # Creates ground truth data frame.
    ground_truth_df = pd.DataFrame({'s-init': gt_s_init,
                                    's-end': gt_s_end,
                                    'video-id': video_id})
    return ground_truth_df

def get_gt_map(gt_file='actNet200-V1-3.pkl'):
    activity_index = {}
    with open(gt_file, 'r') as fobj:
        gt = pickle.load(fobj)['actionIDs']
    for name, anno in gt.iteritems():
        activity_index[anno['class']] = name
    return activity_index

def get_actnet_baseline(top=np.inf, SUBSET='validation', PROPOSALS_FILENAME='activitynet_v1-3_proposals.hdf5'):
    gt_file = 'actNet200-V1-3.pkl'
    with open(gt_file, 'r') as fobj:
        data = pickle.load(fobj)['database']
    intended_videos = []
    for vid, vitem in data.iteritems():
        if vitem['subset'] == SUBSET:
            intended_videos.append('v_{}'.format(vid))

    # Reading proposals from HDF5 file.
    s_init, s_end, score, video_id = [], [], [], []
    fobj = h5py.File(PROPOSALS_FILENAME, 'r')
    for vid in fobj.keys():
        if vid not in intended_videos:
            continue

        starts = fobj[vid]['segment-init'].value.tolist()
        ends = fobj[vid]['segment-end'].value.tolist()
        scs = fobj[vid]['score'].value.tolist()

        s_init.extend(starts[:min(top,len(scs))])
        s_end.extend(ends[:min(top,len(scs))])
        score.extend(scs[:min(top,len(scs))])
        video_id.extend(np.repeat(vid, min(top,len(scs))).tolist())


    fobj.close()
    proposals_df = pd.DataFrame({'s-init': s_init,
                                 's-end': s_end,
                                 'score': score,
                                 'video-id': video_id})
    print 'Average number of proposals: {}'.format(proposals_df.shape[0] / float(len(intended_videos)))
    return proposals_df
