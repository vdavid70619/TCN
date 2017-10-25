import os
import numpy as np
import cPickle as pickle
import tempfile
import h5py
import time
import json
import sys

os.environ['GLOG_minloglevel'] = '2'
import caffe
caffe.set_mode_gpu()
from utils import recall_vs_iou_thresholds, convert, nms, get_gt

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
    layer { name: "data" type: "DummyData" top: "lv" dummy_data_param { shape { dim: 1024 dim: 1 dim: 12 dim: 202 } } }
    layer { name: 'conv' type: 'Convolution' bottom: "lv" top: "conv" convolution_param { engine: CAFFE num_output: 1 kernel_w: 1 kernel_h: 5  weight_filler {type: "xavier"} } }
    layer { name: "relu" type: "ReLU" bottom: "conv" top: "conv" }
    layer { name: "pool" type: "Pooling" bottom: "conv" top: "pool" pooling_param { pool: AVE kernel_h: 3 kernel_w: 1 stride_h: 1 stride_w: 1} }
    #level upper
    layer { name: "data_upper" type: "DummyData" top: "lv_upper" dummy_data_param { shape { dim: 1024 dim: 1 dim: 12 dim: 202 } } }
    layer { name: 'conv_upper' type: 'Convolution' bottom: "lv_upper" top: "conv_upper" convolution_param { engine: CAFFE num_output: 1 kernel_w: 1 kernel_h: 5  weight_filler {type: "xavier"} } }
    layer { name: "relu_upper" type: "ReLU" bottom: "conv_upper" top: "conv_upper" }
    layer { name: "pool_upper" type: "Pooling" bottom: "conv_upper" top: "pool_upper" pooling_param { pool: AVE kernel_h: 3  kernel_w: 1 stride_h: 1 stride_w: 1} }

    layer { name: "concat" type: "Concat" bottom: "pool" bottom: "pool_upper" top: "ip1" concat_param { axis: 1 } }

    #proposal loss
    layer { name: "label" type: "DummyData" top: "label" dummy_data_param { shape { dim: 1024 dim: 1 dim: 1 dim: 1 } } }
    layer { name: "ip1" type: "InnerProduct" bottom: "ip1" top: "ip2" inner_product_param { num_output: 500 weight_filler {type: "xavier"} } }
    layer { name: "ip2" type: "InnerProduct" bottom: "ip2" top: "rs" inner_product_param { num_output: 2 weight_filler {type: "xavier"} } }
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
    layer { name: "data" type: "DummyData" top: "lv" dummy_data_param { shape { dim: 1 dim: 1 dim: 12 dim: 202 } } }
    layer { name: 'conv' type: 'Convolution' bottom: "lv" top: "conv" convolution_param { engine: CAFFE num_output: 1 kernel_w: 1 kernel_h: 5  weight_filler {type: "xavier"} } }
    layer { name: "relu" type: "ReLU" bottom: "conv" top: "conv" }
    layer { name: "pool" type: "Pooling" bottom: "conv" top: "pool" pooling_param { pool: AVE kernel_h: 3 kernel_w: 1 stride_h: 1 stride_w: 1} }
    #level upper
    layer { name: "data_upper" type: "DummyData" top: "lv_upper" dummy_data_param { shape { dim: 1 dim: 1 dim: 12 dim: 202 } } }
    layer { name: 'conv_upper' type: 'Convolution' bottom: "lv_upper" top: "conv_upper" convolution_param { engine: CAFFE num_output: 1 kernel_w: 1 kernel_h: 5  weight_filler {type: "xavier"} } }
    layer { name: "relu_upper" type: "ReLU" bottom: "conv_upper" top: "conv_upper" }
    layer { name: "pool_upper" type: "Pooling" bottom: "conv_upper" top: "pool_upper" pooling_param { pool: AVE kernel_h: 3  kernel_w: 1 stride_h: 1 stride_w: 1} }

    layer { name: "concat" type: "Concat" bottom: "pool" bottom: "pool_upper" top: "ip1" concat_param { axis: 1 } }

    #proposal
    layer { name: "ip1" type: "InnerProduct" bottom: "ip1" top: "ip2" inner_product_param { num_output: 500 } }
    layer { name: "ip2" type: "InnerProduct" bottom: "ip2" top: "rs" inner_product_param { num_output: 2 } }
    layer { name: "loss" type: "Softmax" bottom: "rs" top: "loss"}
    """)
    f.close()
    return f.name

def create_solver(netf):
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write("""
    net: '""" + netf + """'
    base_lr: 0.1
    momentum: 0.9
    weight_decay: 0.00005
    lr_policy: 'step'
    stepsize: 1000
    display: 20000
    snapshot: 1000
    max_iter: 20000
    snapshot_prefix: "rank_pair2"
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
    printout += 'TF {:.2} FF {:.2} \nFB {:.2} TB {:.2} \n'.format(float(tfg)/total, float(ffg)/total, float(fbg)/total, float(tbg)/total, )
    return printout

def extract_feature(feature, proposal, num_frame, duration, PYRAMID=[1,2,4,8,16,16]):
    from tsn import extract
    data = []
    for ind, num in enumerate(PYRAMID):
        feat = extract(feature, num_frame, duration, proposal[ind,0], proposal[ind,1], num, zero_padding=True)
        if feat is None:
            return None
        data.append(feat)
    return np.vstack(data)

# Sample propsoal pairs with a pyramid around specific level (a list) if given
def sample_pairs(proposals, lv=None, scale=2):
    pairs = []
    nlv = proposals.shape[0]

    if lv is not None:
        plist = lv
    else:
        plist = range(nlv)

    for i in plist:
        top = np.zeros((2, 4))
        top[0, :] = proposals[i, :]

        center = (proposals[i, 0] + proposals[i, 1]) / 2
        length = (proposals[i, 1] - proposals[i, 0]) / 2
        top[1, 0] = center - length*scale
        top[1, 1] = center + length*scale
        pairs.append(top)
        #pairs.append(proposals[i:i+2,:])

    return pairs


# Dynamic sample batch size, it is little tricky to do considering a pyramid has n levels.
# Use some hard negatives (all levels 0), the effective number will multiple by n levels
@static_vars(list={})
def load_data_pairs(gt, proposals, batch=1024, random=256, hard_neg=256, maxnump=32, PYRAMID=[12,12], thr_ious=[0.7,0.3], subset='training'):
    x = np.zeros((batch, 1, np.sum(PYRAMID), 202))
    y = np.ones((batch, 1, 1, 1))*-1

    target_pos = batch-hard_neg-random
    pos_count = 0
    neg_count = 0
    rand_count = 0
    count = 0

    if not load_data_pairs.list.has_key(subset):
        # behavior like caffe preload
        load_data_pairs.list[subset] = []
        for vid, vitem in gt.iteritems():
            if vitem['subset'] == subset:
                load_data_pairs.list[subset].append(vid)

    #file_ct = 0
    list = load_data_pairs.list[subset]
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
            pos_pairs += sample_pairs(proposal[ind,], [lv])
        np.random.shuffle(pos_pairs)
        for pair in pos_pairs[:min(target_pos,maxnump)]:
            if pos_count < target_pos and count < batch:
                data = extract_feature(feat, pair, num_frame, duration, PYRAMID=PYRAMID)
                if data is not None and data.shape[0]==np.sum(PYRAMID):
                    x[count, ...].flat = data.flat
                    y[count, ...] = 1
                    pos_count += 1
                    count +=1

        # sample random negative around around positive
        rand_pairs = []
        for i in range(len(pos_indexes)):
            ind = pos_indexes[i]
            lv = max_ious_lv[ind]
            rand_pairs += sample_pairs(proposal[ind,], [i for i in range(6) if i!=lv])
        np.random.shuffle(rand_pairs)
        for pair in rand_pairs[:min(random,maxnump)]:
            if rand_count<random and count < batch:
                data = extract_feature(feat, pair, num_frame, duration, PYRAMID=PYRAMID)
                if data is not None and data.shape[0] == np.sum(PYRAMID):
                    x[count, ...].flat = data.flat
                    y[count, ...] = 0
                    rand_count += 1
                    count += 1

        # sample hard negatives
        neg_pairs = []
        for i in range(len(neg_indexes)):
            ind = neg_indexes[i]
            neg_pairs += sample_pairs(proposal[ind,])
        np.random.shuffle(neg_pairs)
        for pair in neg_pairs[:min(hard_neg,maxnump)]:
            if neg_count < hard_neg and count < batch:
                data = extract_feature(feat, pair, num_frame, duration, PYRAMID=PYRAMID)
                if data is not None and data.shape[0]==np.sum(PYRAMID):
                    x[count, ...].flat = data.flat
                    y[count, ...] = 0
                    neg_count += 1
                    count += 1

        #file_ct += 1
        if count==batch:
            roll = list[pos:] + list[:pos]
            load_data_pairs.list[subset] = roll
            break

    #print 'Files opened %d'%file_ct
    #print subset, neg_count
    #print max_ious
    return x, y

def rank_propsal_pairs(gt, proposals, model, PYRAMID=[12,12], feature_path='/home/DATASETS/actnet/tsn_score/'):
    # load model
    nf = create_deploy()
    sf = create_solver(nf)
    solver = caffe.get_solver(sf)
    solver.net.copy_from(model)

    # classifying
    ACC = []
    ranked_proposals = {}
    for vid, proposal in proposals.iteritems():
        # print vid
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
            current[:,2] = 0
            pairs = sample_pairs(current)
            for pos, pair in enumerate(pairs):
                x = extract_feature(feat, pair, num_frame, duration, PYRAMID=PYRAMID)
                if x is not None and x.shape[0]==np.sum(PYRAMID):
                    x = x[np.newaxis, np.newaxis, ...]

                    solver.net.blobs['lv'].data[...] = x[:,:,0:12,:]
                    solver.net.blobs['lv_upper'].data[...] = x[:,:,12:24,:]

                    rs = solver.net.forward()
                    prob = rs['loss'][0,0,...]
                    ACC.append([[pair[1,3]>0.5],[prob<0.5]])
                    current[pos,2] = 1-prob
                else:
                    current[pos,2] = 0
            ranked_proposals[vid].append(current)

    ACC = np.hstack(ACC)
    print 'Test on validation set ' + stats(ACC[0,:], ACC[1,:])
    return ranked_proposals


if __name__ == '__main__':
    nf = create_net()
    sf = create_solver(nf)
    solver = caffe.get_solver(sf)

    FEATURE = os.environ['ACTNET_HOME'] + '/tsn_score/'

    with open('actNet200-V1-3.pkl', 'rb') as f:
        gt = pickle.load(f)['database']
    with open('train_proposals.pkl', 'rb') as f:
        train_proposals = pickle.load(f)
    with open('val_proposals.pkl', 'rb') as f:
        val_proposals = pickle.load(f)

    max_epoch = 1 #50
    max_iter = 1000 #1000
    print 'Train Proposal Ranker'
    for ep in range(max_epoch):
        loss = 0
        for it in range(max_iter):

            ############# Train #############
            x, y = load_data_pairs(gt, train_proposals)

            solver.net.blobs['lv'].data[...] = x[:, :, 0:12, :]
            solver.net.blobs['lv_upper'].data[...] = x[:, :, 12:24, :]
            solver.net.blobs['label'].data[...] = y
            solver.step(1)
            loss += solver.net.blobs['loss'].data

            if (it+1)%50 == 0:
                print time.strftime('[%Y%-m-%d %H:%M:%S] ') + 'Iter %d '%(max_iter*ep+it+1),
                print 'Train Loss: %f | %d %d'%(loss/50, np.sum(y==0), np.sum(y>=1))
                loss = 0

    model = 'rank_pair2_iter_{}.caffemodel'.format(max_iter*(ep+1))
    ranked_val_proposals = rank_propsal_pairs(gt, val_proposals, model, feature_path=FEATURE)
    with open('ranked_val_proposals.pkl', 'wb') as f:
         pickle.dump(ranked_val_proposals, f)

    proposal_at_1 = {'s-init':[],'s-end':[],'score':[],'label':[],'video-id':[]}
    for vid, proposal in ranked_val_proposals.iteritems():
        proposal = np.asarray(proposal)
        proposal = proposal.reshape((proposal.shape[0]*proposal.shape[1], proposal.shape[2]))
        keep_ind = nms(proposal, proposal[:,2], 0.45)
        proposal = proposal[keep_ind,:]

        proposal_at_1['s-init'].append(proposal[0,0])
        proposal_at_1['s-end'].append(proposal[0,1])
        proposal_at_1['score'].append(proposal[0,2])
        proposal_at_1['video-id'].append('v_'+vid)

    gt = get_gt(gt)
    iou_thrs = np.arange(0.1, 1.0, 0.1)
    recall_at_1 = recall_vs_iou_thresholds(convert(proposal_at_1), gt, iou_threshold=iou_thrs)
    print np.array_str(np.vstack([iou_thrs, recall_at_1]), precision=4, suppress_small=True)

