import os
import numpy as np
import cPickle as pickle
import tempfile
import h5py
import time
import sys
import json

os.environ['GLOG_minloglevel'] = '2'
import caffe
caffe.set_mode_gpu()
from eval_detection import ANETdetection
from utils import gen_json, nms_all, sliding_window_aggregation_func

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def create_net():
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(
    """
    name: 'pair_nn'

    #level base
    layer { name: "data" type: "DummyData" top: "ip1" dummy_data_param { shape { dim: 1024 dim: 1 dim: 1 dim: 40804 } } }

    #proposal loss
    layer { name: "label" type: "DummyData" top: "label" dummy_data_param { shape { dim: 1024 dim: 1 dim: 1 dim: 1 } } }
    layer { name: "ip1" type: "InnerProduct" bottom: "ip1" top: "ip2" inner_product_param { num_output: 2000 weight_filler {type: "xavier"} } }
    layer { name: "ip2" type: "InnerProduct" bottom: "ip2" top: "rs" inner_product_param { num_output: 201 weight_filler {type: "xavier"} } }
    layer { name: "loss" type: "SoftmaxWithLoss" bottom: "rs" bottom: "label" top: "loss" loss_weight: 1 loss_param {ignore_label: -1 normalize: true} }
    """)
    f.close()
    return f.name

def create_deploy():
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(
    """
    name: 'pair_nn'

    #level base
    layer { name: "data" type: "DummyData" top: "ip1" dummy_data_param { shape { dim: 1 dim: 1 dim: 1 dim: 40804 } } }

    #proposal
    layer { name: "ip1" type: "InnerProduct" bottom: "ip1" top: "ip2" inner_product_param { num_output: 2000 } }
    layer { name: "ip2" type: "InnerProduct" bottom: "ip2" top: "rs" inner_product_param { num_output: 201 } }
    layer { name: "loss" type: "Softmax" bottom: "rs" top: "prob"}
    """)
    f.close()
    return f.name

def create_solver(netf):
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write("""
    net: '""" + netf + """'
    base_lr: 0.0001
    momentum: 0.9
    weight_decay: 0.00005
    lr_policy: 'step'
    stepsize: 1000
    display: 20000
    snapshot: 1000
    max_iter: 20000
    snapshot_prefix: "class_bl2"
    """)
    f.close()
    return f.name

def stats(label, pred, hist=False):
    printout = ''
    if hist==True:
        y = label.squeeze()
        ind = np.nonzero(y > 0)
        neg = np.nonzero(y == 0)
        _, ct = np.unique(ind[y.ndim-1], return_counts=True)
        printout += 'Hist ({}|{}) '.format(len(neg[y.ndim-1]), ct)

    fg = np.sum(label >= 1)
    bg = np.sum(label == 0)
    total = fg+bg
    corr = np.sum(label == pred)
    tfg = np.sum(np.logical_and(label>=1,pred>=1))
    tbg = np.sum(np.logical_and(label==0,pred==0))
    ffg = np.sum(np.logical_and(label==0,pred>=1))
    fbg = np.sum(np.logical_and(label>=1,pred==0))
    printout += 'Acc: {:.4} \n'.format(float(corr)/(fg+bg))
    printout += 'TF {:.2} FF {:.2} \nFB {:.2} TB {:.2} \n'.format(float(tfg)/fg, float(ffg)/bg, float(fbg)/fg, float(tbg)/bg, )
    return printout

def extract_feature(feature, proposal, num_frame, duration, PYRAMID=[10000]):
    from tsn import extract
    data = []
    for ind, num in enumerate(PYRAMID):
        feat = extract(feature, num_frame, duration, proposal[ind,0], proposal[ind,1], num, repeat=False)
        if feat is None:
            return None
        data.append(feat)
    return np.vstack(data)


# Dynamic sample batch size, it is little tricky to do considering a pyramid has n levels.
# Use some hard negatives (all levels 0), the effective number will multiple by n levels
@static_vars(list={})
def load_data(gt, proposals, batch=1024, random=32, hard_neg=32, maxnump=32, thr_ious=[0.7,0.3], subset='training'):
    x = np.zeros((batch, 1, 1, 202*202))
    y = np.ones((batch, 1, 1, 1))*-1

    target_pos = batch-hard_neg-random
    pos_count = 0
    neg_count = 0
    rand_count = 0
    count = 0

    if not load_data.list.has_key(subset):
        # behavior like caffe preload
        load_data.list[subset] = []
        for vid, vitem in gt.iteritems():
            if vitem['subset'] == subset:
                load_data.list[subset].append(vid)

    #file_ct = 0
    list = load_data.list[subset]
    for pos, vid in enumerate(list):
        vitem = gt[vid]

        if not os.path.exists(FEATURE + '/feat/%s.h5' % vid) or not proposals.has_key(vid) or len(proposals[vid])==0:
            continue

        with h5py.File(FEATURE + '/feat/%s.h5' % vid, 'r') as hf:
            fg = np.asarray(hf['fg'])
            bg = np.asarray(hf['bg'])
            feat = np.hstack([fg,bg])
        with h5py.File(FEATURE + '/flow/%s.h5' % vid, 'r') as hf:
            fg2 = np.asarray(hf['fg'])
            bg2 = np.asarray(hf['bg'])
            feat2 = np.hstack([fg2,bg2])
        feat = feat + feat2

        num_frame = vitem['numf']
        duration = vitem['duration']
        proposal = np.asarray(proposals[vid])

        max_ious = proposal[:, :, 2].max(axis=1)
        max_ious_lv = proposal[:, :, 2].argmax(axis=1)
        #positive per level
        pos_indexes = np.nonzero(max_ious > thr_ious[0])[0]
        #hard negative
        neg_indexes = np.nonzero(max_ious < thr_ious[1])[0]


        # sample postives around the best level
        pos_pairs = []
        for i in range(len(pos_indexes)):
            ind = pos_indexes[i]
            lv = max_ious_lv[ind]
            pos_pairs += [proposal[ind,lv:lv+1,:]]
        np.random.shuffle(pos_pairs)
        for pair in pos_pairs[:min(target_pos,maxnump)]:
            if pos_count < target_pos and count < batch:
                data = extract_feature(feat, pair, num_frame, duration)
                if data is not None:
                    x[count, ...].flat = tempo_bilinear(data).flat
                    y[count, ...] = pair[0,3]
                    pos_count += 1
                    count +=1

        # sample random negative around around positive
        rand_pairs = []
        for i in range(len(pos_indexes)):
            ind = pos_indexes[i]
            lv = max_ious_lv[ind]
            rand_pairs += [proposal[ind,i:i+1,] for i in range(6) if i!=lv]
        np.random.shuffle(rand_pairs)
        for pair in rand_pairs[:min(random,maxnump)]:
            if rand_count<random and count < batch:
                data = extract_feature(feat, pair, num_frame, duration)
                if data is not None:
                    x[count, ...].flat = tempo_bilinear(data).flat
                    y[count, ...] = pair[0,3]
                    rand_count += 1
                    count += 1

        # sample hard negatives
        neg_pairs = []
        for i in range(len(neg_indexes)):
            ind = neg_indexes[i]
            neg_pairs += [proposal[ind,i:i+1,] for i in range(6)]
        np.random.shuffle(neg_pairs)
        for pair in neg_pairs[:min(hard_neg,maxnump)]:
            if neg_count < hard_neg and count < batch:
                data = extract_feature(feat, pair, num_frame, duration)
                if data is not None:
                    x[count, ...].flat = tempo_bilinear(data).flat
                    y[count, ...] = 0
                    neg_count += 1
                    count += 1

        #file_ct += 1
        if count==batch:
            roll = list[pos:] + list[:pos]
            load_data.list[subset] = roll
            break

    #print 'Files opened %d'%file_ct
    #print subset, neg_count
    #print max_ious
    return x, y

def tempo_bilinear(feat):
    return np.dot(feat.T, feat)

def classify_proposal(gt, proposals, model, feature_path='/home/DATASETS/actnet/tsn_score/', priors=[], weights=[], topK=1, verbose=False):
    # load model
    nf = create_deploy()
    net = caffe.Net(nf, model, caffe.TEST)

    # classifying
    ACC = []
    ranked_proposals = {}
    for vid, proposal in proposals.iteritems():
        if not os.path.exists(feature_path + '/feat/%s.h5' % vid) or len(proposals)==0:
            continue

        with h5py.File(feature_path + '/feat/%s.h5' % vid, 'r') as hf:
            fg = np.asarray(hf['fg'])
            bg = np.asarray(hf['bg'])
            feat = np.hstack([fg,bg])
        with h5py.File(feature_path + '/flow/%s.h5' % vid, 'r') as hf:
            fg2 = np.asarray(hf['fg'])
            bg2 = np.asarray(hf['bg'])
            feat2 = np.hstack([fg2,bg2])
        feat = feat + feat2

        num_frame = gt[vid]['numf']
        duration = gt[vid]['duration']
        proposal = np.asarray(proposal)
        ranked_proposals[vid] = []
        for i in range(proposal.shape[0]):
            current = np.copy(proposal[i,...])
            current[3] = 0
            x = extract_feature(feat, current[np.newaxis,:], num_frame, duration)
            if x is not None:
                net.blobs['ip1'].data[...] = tempo_bilinear(x).flat
                rs = net.forward()
                prob = rs['prob'].squeeze()
                for prior, weight in zip(priors, weights):
                    prob += prior[vid].squeeze()*weight
                pred = prob.argmax()
                ACC.append([[proposal[i,3]],[pred]])
                current[3] = pred

            # print current
            ranked_proposals[vid].append(current)

    if verbose:
        ACC = np.hstack(ACC)
        print 'Test on validation set ' + stats(ACC[0,:], ACC[1,:])
    return ranked_proposals

def classify_video(gt, feature_path='/home/DATASETS/actnet/tsn_score', subset='validation'):

    score_per_video = {}
    for vid, vitem in gt.iteritems():
        if vitem['subset'] != subset or not os.path.exists(feature_path + '/feat/%s.h5' % vid):
            continue

        with h5py.File(feature_path + '/feat/%s.h5' % vid, 'r') as hf:
            fg = np.asarray(hf['fg'])
        with h5py.File(feature_path + '/flow/%s.h5' % vid, 'r') as hf:
            fg += np.asarray(hf['fg'])

        model_scores = sliding_window_aggregation_func(fg[:,np.newaxis,:], norm=False)
        model_scores = np.hstack((0, model_scores))
        score_per_video[vid] = model_scores

    return score_per_video


if __name__ == '__main__':
    nf = create_net()
    sf = create_solver(nf)
    solver = caffe.get_solver(sf)

    FEATURE = os.environ['ACTNET_HOME'] + '/tsn_score/'

    with open('actNet200-V1-3.pkl', 'rb') as f:
        gt = pickle.load(f)
        names = gt['actionIDs']
        gt = gt['database']

    with open('train_proposals.pkl', 'rb') as f:
        train_proposals = pickle.load(f)

    with open('ranked_val_proposals.pkl', 'rb') as f:
        ranked_val_proposals = pickle.load(f)


    max_epoch = 1 #50
    max_iter = 1000 #1000
    print 'Train Proposal Classifier'
    for ep in range(max_epoch):
        loss = 0
        #for it in range(max_iter):
        for it in range(max_iter):

            ############# Train #############
            x, y = load_data(gt, train_proposals)

            solver.net.blobs['ip1'].data[...] = x
            solver.net.blobs['label'].data[...] = y
            solver.step(1)
            loss += solver.net.blobs['loss'].data

            if (it+1)%50 == 0:
                print time.strftime('[%Y%-m-%d %H:%M:%S] ') + 'Iter %d '%(max_iter*ep+it+1),
                print 'Train Loss: %f | %d %d'%(loss/50, np.sum(y==0), np.sum(y>=1))
                loss = 0

        model = 'class_bl2_iter_{}.caffemodel'.format(max_iter*(ep+1))

    ranked_val_proposals = nms_all(ranked_val_proposals, topK=20, nms_thor=0.45)

    score_tsn = classify_video(gt, subset='validation', feature_path=FEATURE)
    classified_proposals = classify_proposal(gt, ranked_val_proposals, model, feature_path=FEATURE,
                                             priors=[score_tsn], weights=[1],
                                             )

    classified_proposals = nms_all(classified_proposals, topK=20, nms_thor=1, remove_background=True)

    # test on offical evaluation code
    id2name = {}
    for name, ids in names.iteritems():
        id2name[ids['class']] = name
    output = gen_json(classified_proposals, id2name)
    with open('prediction.json', 'w') as f:
        json.dump(output, f)
    eval1 = ANETdetection('activity_net.v1-3.min.json', 'prediction.json', subset='validation',
                              tiou_thr=0.5, verbose=True, check_status=False)
    eval1.evaluate()
    eval2 = ANETdetection('activity_net.v1-3.min.json', 'prediction.json', subset='validation',
                              tiou_thr=0.75, verbose=True, check_status=False)
    eval2.evaluate()
    eval3 = ANETdetection('activity_net.v1-3.min.json', 'prediction.json', subset='validation',
                              tiou_thr=0.95, verbose=True, check_status=False)
    eval3.evaluate()
