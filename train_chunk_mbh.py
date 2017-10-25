import h5py
import pickle
import numpy as np
import tempfile
import time
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
caffe.set_mode_gpu()

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def create_fg_net():
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(
    """
    name: 'nn'
    layer { name: "data" type: "DummyData" top: "data" dummy_data_param { shape { dim: 1 dim: 1 dim: 1 dim: 65536 } } }
    layer { name: "label" type: "DummyData" top: "label" dummy_data_param { shape { dim: 1 dim: 1 dim: 1 dim: 1 } } }

    layer { name: 'ip' type: 'InnerProduct' bottom: 'data' top: 'ip1' inner_product_param { num_output: 2000  weight_filler {type: "xavier"} } }
    layer { name: 'drop' type: 'Dropout' bottom: 'ip1' top: 'ip1' dropout_param { dropout_ratio: 0.5 } }
    layer { name: 'ip2' type: 'InnerProduct' bottom: 'ip1' top: 'ip2' inner_product_param { num_output: 201  weight_filler {type: "xavier"} } }
    layer { name: 'loss' type: 'SoftmaxWithLoss' bottom: 'ip2' bottom: 'label' top: 'loss' loss_param { normalize: true ignore_label: 0} }
    """)
    f.close()
    return f.name

def create_solver(netf):
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write("""
    net: '""" + netf + """'
    base_lr: 1
    momentum: 0.9
    weight_decay: 0.00005
    lr_policy: 'fixed'
    display: 200000
    snapshot: 1000
    max_iter: 200000
    snapshot_prefix: "chunk_mbh"
    """)
    f.close()
    return f.name


def create_deploy():
    f = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    f.write(
    """
    name: 'nn'
    layer { name: "data" type: "DummyData" top: "data" dummy_data_param { shape { dim: 1 dim: 1 dim: 1 dim: 65536 } } }

    layer { name: 'ip' type: 'InnerProduct' bottom: 'data' top: 'ip1' inner_product_param { num_output: 2000  weight_filler {type: "xavier"} } }
    layer { name: 'ip2' type: 'InnerProduct' bottom: 'ip1' top: 'ip2' inner_product_param { num_output: 201  weight_filler {type: "xavier"} } }
    layer { name: 'loss' type: 'Softmax' bottom: 'ip2' top: 'prob' }
    """)
    f.close()
    return f.name


def classify_video(model, subset='validation', ANNOTATION = 'actNet200-V1-3.pkl', FEATURE = '/home/EXTRA/DATASETS/actnet/mbh/'):
    nf = create_deploy()
    net = caffe.Net(nf, model, caffe.TEST)

    with open(ANNOTATION, 'rb') as f:
        gt = pickle.load(f)['database']
    with h5py.File(FEATURE + 'MBH_Videos_features.h5', 'r') as f:
        mbh = np.asarray(f['features'])
    with open(FEATURE + 'MBH_Videos_quids.txt', 'r') as f:
        lines = f.readlines()
        idmap = {}
        for line in lines:
            temp = line.split(',')
            idmap[temp[2].split('.')[0]] = int(temp[3].replace(')\n',''))

    scores = {}
    for vid, vitem in gt.iteritems():
        if vitem['subset']==subset:
            ind = idmap['v_'+vid]
            feat = mbh[ind, :]

            net.blobs['data'].data[...] = feat
            net.forward()

            prob = net.blobs['prob'].data
            scores[vid] = np.copy(prob)
    return scores

if __name__ == '__main__':

    import logging
    logging.basicConfig(format='%(asctime)s %(message)s', filename='train_chunk_mbh'+time.strftime('_%Y_%m_%d.log'), level=logging.DEBUG)
    logging.getLogger()
    logging.info(' 201 classifier with mbh')

    FEATURE = os.environ['ACTNET_HOME'] + '/mbh/'

    with open('actNet200-V1-3.pkl', 'rb') as f:
        gt = pickle.load(f)['database']
    with h5py.File(FEATURE + 'MBH_Videos_features.h5', 'r') as f:
        mbh = np.asarray(f['features'])
    with open(FEATURE + 'MBH_Videos_quids.txt', 'r') as f:
        lines = f.readlines()
        idmap = {}
        for line in lines:
            temp = line.split(',')
            idmap[temp[2].split('.')[0]] = temp[3].replace(')\n','')

    indexes = {'training':[],'validation':[],'testing':[]}
    labels = {'training':[],'validation':[]}
    for vid, vitem in gt.iteritems():
        if vitem['subset']!='testing':
            labels[vitem['subset']].append(vitem['annotations'][0]['class'])
        indexes[vitem['subset']].append(idmap['v_'+vid])

    ############# Train and test a nn #############
    nf = create_fg_net()
    sf = create_solver(nf)
    solver = caffe.get_solver(sf)

    logging.debug('')

    max_ep = 20
    max_iter = 50
    batch = 200 
    
    # test_feat = mbh[np.array(indexes['testing'], dtype=int),:]
    list = indexes['training']
    for ep in range(max_ep):
        logging.info('Epoch %d ' % (ep + 1))
        loss = 0
        for it in range(max_iter):
            ############# Train #############
            x = mbh[np.array(indexes['training'][it*batch:(it+1)*batch], dtype=int),:]
            y = np.array(labels['training'][it*batch:(it+1)*batch], dtype=int)

            x = x[:, np.newaxis, np.newaxis, :]
            y = y[:, np.newaxis, np.newaxis, np.newaxis]

            solver.net.blobs['data'].reshape(*x.shape)
            solver.net.blobs['data'].data[...] = x
            solver.net.blobs['label'].reshape(*y.shape)
            solver.net.blobs['label'].data[...] = y
            solver.step(1)
            loss += solver.net.blobs['loss'].data
        logging.info('Train Loss: %f '%(loss/max_iter))
        ############# Test #############

        val_feat = mbh[np.array(indexes['validation'], dtype=int), :]
        val_lb = np.array(labels['validation'])
        val_feat = val_feat[:, np.newaxis, np.newaxis, :]
        val_lb = val_lb[:, np.newaxis, np.newaxis, np.newaxis]


        solver.net.blobs['data'].reshape(*val_feat.shape)
        solver.net.blobs['data'].data[...] = val_feat
        solver.net.blobs['label'].reshape(*val_lb.shape)
        solver.net.forward()

        prob = solver.net.blobs['ip2'].data
        pred = prob.argmax(axis=1)

        logging.info('Test on validation set ACC %f'%(np.sum(pred.squeeze()==val_lb.squeeze())*1.0/val_lb.shape[0]))
